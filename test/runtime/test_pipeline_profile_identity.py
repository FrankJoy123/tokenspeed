# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Profile files identify actual ranks across every pipeline stage."""

import pytest

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.engine.profiler import _profile_rank_tag


@pytest.mark.parametrize("pp_size", [1, 2, 4, 8])
def test_profile_names_do_not_collide_across_pipeline_stages(pp_size):
    tags = set()
    for rank in range(8):
        mapping = Mapping(world_size=8, pp_size=pp_size, attn_tp_size=8 // pp_size)
        mapping.rank = rank
        tag = _profile_rank_tag(
            mapping.attn, pp_rank=mapping.pp_rank if pp_size > 1 else None
        )
        assert tag not in tags
        tags.add(tag)
    assert len(tags) == 8
