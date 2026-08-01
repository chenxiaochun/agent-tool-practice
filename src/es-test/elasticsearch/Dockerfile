# Elasticsearch 8.17.0 + IK 中文分词（构建时打进镜像）
# 使用 Docker Hub 官方镜像，避免 docker.elastic.co 在国内过慢
FROM elasticsearch:8.17.0

# --batch：非交互安装；版本必须与 ES 完全一致
RUN elasticsearch-plugin install --batch \
  https://release.infinilabs.com/analysis-ik/stable/elasticsearch-analysis-ik-8.17.0.zip
