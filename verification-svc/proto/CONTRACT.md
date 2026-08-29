# Upstream Contract Assumptions (Member 3: Coarse Matching)

**STATUS: UNVERIFIED**
This contract is based on the shared mental model and the Member 3 master prompt. It has NOT been verified against a real implementation of `coarse-matching-svc` as that service does not yet exist in this repository.

## 1. Candidate Matches (MatchResult)
The `candidate_matches_ref` points to a Parquet file with the following schema:

| Field | Type | Description |
| :--- | :--- | :--- |
| `p_x` | float | X coordinate in source image |
| `p_y` | float | Y coordinate in source image |
| `q_x` | float | X coordinate in reference basemap |
| `q_y` | float | Y coordinate in reference basemap |
| `source_method` | string | Origin of match (e.g., 'classical', 'deep') |
| `confidence` | float | Confidence score of the match |
| `pyramid_level` | int | The scale level where the match was found |

## 2. RemineTile RPC
**Request (`RemineRequest`):**
- `job_id` (string): Unique job identifier.
- `tile_id` (int): Index of the under-covered tile.
- `tile_bounds` (float[4]): `[xmin, ymin, xmax, ymax]` of the tile in the reference image.
- `relaxed_confidence_threshold` (float): Lowered threshold to increase recall.

**Response (`RemineResult`):**
- `additional_candidates_ref` (string): S3 URI to a Parquet file containing new candidate matches for that specific tile.

## Reconciliation Note
Any changes made by Member 3 to the `coarse-matching-svc` API or data schema must be coordinated with this service to avoid pipeline breakage.
