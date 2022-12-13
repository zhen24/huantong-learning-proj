# huantong-learning-proj

```bash
rm -f data/wheels/huantong
python setup.py clean --all
python -m build --wheel

LATEST_WHEEL=$(ls -t dist/*-py3-none-any.whl | head -1 | xargs -n 1 basename)

docker run \
  --rm -it \
  --user "$(id -u):$(id -g)" \
  -e PYTHON_ABI_TAG=cp38-cp38 \
  -v "$(pwd)":/data \
  pywhlobf/pywhlobf:0.1.4-manylinux2014_x86_64 \
  "/data/dist/${LATEST_WHEEL}" \
  '/data/dist/pywhlobf'

OBF_WHEEL=$(ls -t dist/pywhlobf/*-cp38-cp38-*.whl | head -1 | xargs -n 1 basename)

chmod -R 755 dist/
cp "dist/pywhlobf/${OBF_WHEEL}" "data/wheels/"

python setup.py clean --all
rm -rf huantong_learning_proj.egg-info/ dist/
```

# build and save image

```bash
docker build \
    --network host \
    --build-arg http_proxy=http://192.168.0.153:8001 \
    --build-arg https_proxy=http://192.168.0.153:8001 \
    --build-arg HTTP_PROXY=http://192.168.0.153:8001 \
    --build-arg HTTPS_PROXY=http://192.168.0.153:8001 \
    -t huantong_learning_proj/api:0.1.0 \
    -f dockerfile/api \
    .


docker build \
    --network host \
    --build-arg http_proxy=http://192.168.0.153:8001 \
    --build-arg https_proxy=http://192.168.0.153:8001 \
    --build-arg HTTP_PROXY=http://192.168.0.153:8001 \
    --build-arg HTTPS_PROXY=http://192.168.0.153:8001 \
    -t huantong_learning_proj/job:0.1.0 \
    -f dockerfile/job \
    .


docker save huantong_learning_proj/api:0.1.0 | pigz > /home/mnvai/projects/huantong/image/huantong_learning_proj-api-0.1.0.tgz
docker save huantong_learning_proj/job:0.1.0 | pigz > /home/mnvai/projects/huantong/image/huantong_learning_proj-job-0.1.0.tgz
# docker cp huantong-learning-trainer:/usr/local/lib/python3.8/dist-packages/huantong_learning_proj/job.py ./
```
