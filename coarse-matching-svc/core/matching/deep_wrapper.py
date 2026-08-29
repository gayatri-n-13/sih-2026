import numpy as np
from typing import List, Tuple
from core.matching.base import Matcher
from models.backbone.interface import Backbone
from models.data_models import Correspondence, Transform

class DeepMatcher(Matcher):
    """
    Wrapper for deep learned matchers. Implements tiling for large images.
    """
    def __init__(self, backbone: Backbone, tile_size: int = 640, overlap: int = 64):
        self.backbone = backbone
        self.tile_size = tile_size
        self.overlap = overlap

    def _get_tiles(self, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Calculates tile boundaries (y0, y1, x0, x1)."""
        tiles = []
        stride = self.tile_size - self.overlap

        for y0 in range(0, h, stride):
            y1 = min(y0 + self.tile_size, h)
            # Adjust y0 if we are at the end to keep tile_size
            if y1 == h: y0 = max(0, h - self.tile_size)

            for x0 in range(0, w, stride):
                x1 = min(x0 + self.tile_size, w)
                if x1 == w: x0 = max(0, w - self.tile_size)
                tiles.append((y0, y1, x0, x1))
        return tiles

    def match(self, img_ref: np.ndarray, img_src: np.ndarray,
              initial_transform: Transform) -> List[Correspondence]:
        """
        Tiled matching using the deep backbone.
        """
        h, w = img_ref.shape
        tiles = self._get_tiles(h, w)
        all_matches = []

        for (y0, y1, x0, x1) in tiles:
            patch_ref = img_ref[y0:y1, x0:x1]
            patch_src = img_src[y0:y1, x0:x1]

            # Deep backbone returns local matches
            local_matches = self.backbone.match(patch_ref, patch_src)

            for p_ref, p_src, conf in local_matches:
                # Convert local coordinates to global
                global_ref = (p_ref[0] + x0, p_ref[1] + y0)
                global_src = (p_src[0] + x0, p_src[1] + y0)

                all_matches.append(Correspondence(
                    pt_ref=global_ref,
                    pt_src=global_src,
                    confidence=conf,
                    source="deep"
                ))

        return all_matches
