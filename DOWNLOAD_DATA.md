# Download all data from GCP bucket

$BUCKET = "322793365"

# Create directories
New-Item -ItemType Directory -Force -Path "data/postings_gcp/text"
New-Item -ItemType Directory -Force -Path "data/postings_gcp/title"
New-Item -ItemType Directory -Force -Path "data/postings_gcp/anchor"
New-Item -ItemType Directory -Force -Path "data/pr"
New-Item -ItemType Directory -Force -Path "data/pv"
New-Item -ItemType Directory -Force -Path "data/mappings"
New-Item -ItemType Directory -Force -Path "data/embeddings/title"

Write-Host "`n📦 Downloading text index..."
gsutil -m cp "gs://$BUCKET/322793365/postings_gcp/text/*.bin" "data/postings_gcp/text/"
gsutil cp "gs://$BUCKET/322793365/postings_gcp/text/text_index.pkl" "data/postings_gcp/text/"

Write-Host "`n📦 Downloading title index..."
gsutil -m cp "gs://$BUCKET/322793365/postings_gcp/title/*.bin" "data/postings_gcp/title/"
gsutil cp "gs://$BUCKET/322793365/postings_gcp/title/title_index.pkl" "data/postings_gcp/title/"

Write-Host "`n📦 Downloading anchor index..."
gsutil -m cp "gs://$BUCKET/322793365/postings_gcp/anchor/*.bin" "data/postings_gcp/anchor/"
gsutil cp "gs://$BUCKET/322793365/postings_gcp/anchor/anchor_index.pkl" "data/postings_gcp/anchor/"

Write-Host "`n📦 Downloading PageRank..."
gsutil -m cp "gs://$BUCKET/pr/*.csv.gz" "data/pr/"

Write-Host "`n📦 Downloading page views..."
gsutil cp "gs://$BUCKET/pv/pageview.pkl" "data/pv/"

Write-Host "`n📦 Downloading title mappings..."
gsutil cp "gs://$BUCKET/mappings/doc_id_to_title.pkl" "data/mappings/"

Write-Host "`n📦 Downloading embeddings..."
gsutil cp "gs://$BUCKET/embeddings/title/*.npy" "data/embeddings/title/"

Write-Host "`n✅ Download complete!"
Write-Host "=" * 60