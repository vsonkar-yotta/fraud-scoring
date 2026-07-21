FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# local corporate CA bundle (network intercepts pypi.org); not committed, .gitignore'd
COPY ca-bundle.pem /usr/local/share/ca-certificates/local-ca.crt
RUN update-ca-certificates
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir .

# model baked into the image at build time; override to ship a different
# promoted version without changing code
ARG MODEL_VERSION=current
COPY configs/serve.yaml configs/serve.yaml
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "fraud_scoring.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
