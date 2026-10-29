# SPDX-License-Identifier: Apache-2.0
"""MMuse-specific ACL graph capture and current-step attention updates."""

from contextlib import contextmanager

import torch
from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.worker.gpu.spec_decode.mmuse.greedy import MMuseGraphManager, MMuseGreedySpeculator

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import set_draft_graph_params
from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph
from vllm_ascend.utils import use_updatable_graph
from vllm_ascend.worker.v2.aclgraph_utils import collect_sorted_captured_token_sizes
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata, build_attn_metadata_wrapper
from vllm_ascend.worker.v2.spec_decode.pcp_utils import disable_profiling_chunk_for_draft
from vllm_ascend.worker.v2.utils import communicator_switch


class MMuseAclGraphManager(MMuseGraphManager):
    def capture_mmuse(self, proposer):
        self.proposer = proposer
        if self.needs_capture():
            layers = get_layers_from_vllm_config(
                proposer.vllm_config, AttentionLayerBase, proposer.draft_attn_layer_names
            )
            if not layers or any(not use_updatable_graph(layer.get_attn_backend()) for layer in layers.values()):
                raise ValueError("MMuse FULL currently requires the Ascend updatable attention backend")
            set_draft_graph_params(collect_sorted_captured_token_sizes(self._capture_descs))
        with communicator_switch():
            super().capture_mmuse(proposer)

    def run_fullgraph(self, desc):
        graph = self.graphs[desc]
        if not isinstance(graph, UpdatableGraph):
            raise TypeError("MMuse ACL replay requires UpdatableGraph")
        # Resolve against the metadata already built for this step. Do not
        # rebuild it, or use metadata retained from a previous replay.
        tasks = graph.resolve_tasks(ContextSource(self.proposer.replay_metadata))
        stream = self.proposer.update_stream
        stream.wait_stream(torch.npu.current_stream())
        result = super().run_fullgraph(desc)
        graph.update(stream, tasks)
        return result


class AscendMMuseGreedySpeculator(MMuseGreedySpeculator):
    graph_manager_cls = MMuseAclGraphManager
    capture_metadata = staticmethod(build_attn_metadata)

    def load_draft_model(self, target_model, target_attn_layer_names):
        with disable_profiling_chunk_for_draft(self.vllm_config):
            return super().load_draft_model(target_model, target_attn_layer_names)

    def set_attn(self, *args, **kwargs):
        super().set_attn(*args, **kwargs)
        self.context_slots = {gid: slots.to(torch.int32) for gid, slots in self.context_slots.items()}

    def build_query_metadata(self, batch, desc):
        with build_attn_metadata_wrapper():
            return super().build_query_metadata(batch, desc)

    def fix_query_lengths(self, metadata, count):
        lengths = [(i + 1) * self.num_query_per_req for i in range(count)]
        for layer in metadata.values():
            layer.actual_seq_lengths_q = lengths

    @contextmanager
    def query_context(self):
        previous = (_EXTRA_CTX.capturing, _EXTRA_CTX.is_draft_model, _EXTRA_CTX.is_draft_model_prefill)
        _EXTRA_CTX.capturing = torch.npu.is_current_stream_capturing()
        _EXTRA_CTX.is_draft_model = True
        _EXTRA_CTX.is_draft_model_prefill = False
        try:
            yield
        finally:
            (_EXTRA_CTX.capturing, _EXTRA_CTX.is_draft_model, _EXTRA_CTX.is_draft_model_prefill) = previous
