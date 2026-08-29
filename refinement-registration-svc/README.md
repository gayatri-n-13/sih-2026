# Sub-Pixel Refinement & Geometric Registration Service

This microservice is part of the lunar image registration pipeline. It takes coarsely verified correspondences and refines them to sub-pixel accuracy to produce a final geometrically registered image.

## Architecture

The service consists of three main modules:
- `core.refinement`: Implements phase correlation and Least-Squares Matching (LSM) for sub-pixel point refinement.
- `core.transform`: Handles selection of the geometric transform model (Similarity, Affine, Projective) and robust fitting using IRLS/Huber loss.
- `core.registration`: Performs final image resampling using high-quality interpolation and writes the output as a Cloud-Optimized GeoTIFF (COG).

## Implementation Details

### Sub-Pixel Refinement
The service uses a hybrid approach:
1. **Phase Correlation**: Provides a coarse sub-pixel shift estimate.
2. **Least-Squares Matching (LSM)**: Jointly optimizes for geometric shift and radiometric gain/bias to achieve scientific-grade accuracy.

### Transform Fitting
Robust fitting is achieved via the `scipy.optimize.least_squares` implementation of the Huber loss, ensuring that residual outliers in the match pool do not skew the final registration.

## Setup and Running

### Prerequisites
- Python 3.11+
- GDAL installed on the system

### Installation
```bash
pip install -r requirements.txt
```

### Running the Service
```bash
python refinement_registration_svc/api/server.py
```

## API Contract

### `POST /register`
Submits a registration job.
**Request**: `RegisterRequest` (see `api/server.py`)
**Response**: `job_id`

### `GET /status/{job_id}`
Checks job status and retrieves results.
**Response**: `RegisterResult`

## Testing
Run the test suite:
```bash
pytest refinement_registration_svc/tests/
```
The `test_refinement.py` includes a hard-gate test for sub-pixel shift recovery.
