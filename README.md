

```sh
playwright install
playwright install-deps
```


```sh
git clone git@github.com:wonchul-kim/similarity_inspection.git
cd similarity_inspection
docker build -t copyright .
docker run -it --name copyright --ipc host --gpus all -v <current absolute path>:/workspace copyright bash
```
```sh
apt-get update 
apt-get install -y libgl1
apt-get install -y libglib2.0-0
```

```sh
pip inestall -e .
pip install -r requiremets.txt
```

```sh
python3 copyright/build_index.py
```


```sh
python3 copyright/scan.py
```
