# Geometric Verification & Uniform-Coverage Service

This microservice is part of the Chandrayaan-2 lunar image registration pipeline. Its primary goal is to convert a noisy pool of candidate correspondences into a robust, geometrically consistent, and spatially uniform set of inlier matches.

## Core Logic

The service follows a multi-stage process:
1. **Robust Estimation**: Uses `cv2.estimateAffine2D` with RANSAC (and preference for MAGSAC++) to find a global geometric model.
2. **Local Consistency**: Re-verifies inliers against a locally-linearized model to handle high-relief terrain.
3. **Uniform Coverage Audit**:
    - Overlays a $K \times K$ grid on the image.
    - Applies Non-Maximum Suppression (NMS) within each tile to prevent clustering.
    - Identifies "under-covered" tiles with fewer than $m_{min}$ matches.
4. **Re-mining Loop**: Calls the `CoarseMatchingService.RemineTile` RPC for under-covered regions, iteratively improving coverage until a budget is reached.

## API Contract

### gRPC Interface
- `Verify(VerifyRequest) -> JobHandle`: Starts the verification process.
- `GetVerifyStatus(JobHandle) -> VerifyResult`: Polls for the result, which includes the verified match set and a detailed `CoverageReport`.

### Upstream Dependency
The service is a client of the `CoarseMatchingService`:
- `RemineTile(job_id, tile_id, tile_bounds, relaxed_threshold) -> additional_candidates_ref`

## Configuration

The following parameters are configurable via the `config_ref`:
- `tile_grid_rows` / `tile_grid_cols`: Dimensions of the coverage grid (default 8x8).
- `m_min`: Minimum required matches per tile to be considered "covered".
- `m_max`: Maximum matches to keep per tile (NMS).
- `remine_budget`: Maximum number of re-mining iterations.
- `relaxed_threshold`: Confidence threshold used for re-mining calls.

## Running the Service

### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   python src/verification_svc/server.py
   ```
3. Run tests:
   ```bash
   pytest tests/
   ```

### Docker
```bash
docker-compose up --build
```

## Mocking for Testing
The service includes a `CoarseMatchingMock` in `src/verification_svc/mocks.py` which simulates the S3 storage and the `RemineTile` RPC, allowing the verification service to be developed and tested in isolation.
