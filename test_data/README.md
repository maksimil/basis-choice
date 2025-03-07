# test_data

matrices for test are converted to`.mtx` format from linear constraints of
[QP-Test-Problems](https://github.com/YimingYAN/QP-Test-Problems) (in
`QPS_Files`) using `./extract-constraints.py` and `./convert.sh`.

## Usage of scripts

Extract linear constraints from `file.QPS` to `file.mtx`.
```
python3 extract-constraints.py file.QPS file.mtx
```

Extract linear constraints from all `.qps` files in `folder` and write `.mtx`
files in the current working directory.
```
sh convert.sh folder
```

