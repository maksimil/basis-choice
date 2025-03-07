for file in $1/*.QPS
do
    echo "python3 ./extract-constraints.py $file ./$(basename -s .QPS $file).mtx"
    python3 ./extract-constraints.py $file ./$(basename -s .QPS $file).mtx
done
