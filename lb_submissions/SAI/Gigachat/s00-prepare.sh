set -e
sogma_xml_file="sogma-test.xml"
sogma_jsonl_file="RuMedTest--sogma--dev.jsonl"
medbench_url="https://medbench.ru/files/MedBench_data.zip"
medbench_dir="MedBench"
medbench_zip_path="$medbench_dir/MedBench_data.zip"

if [ ! -e "$sogma_jsonl_file" ]; then
    echo "$sogma_jsonl_file does not exist. Downloading..."
    wget -nc "https://geetest.ru/content/files/terapiya_(dlya_internov)_sogma_.xml" -O "$sogma_xml_file"
    echo "Download complete."
    python convert_sogma.py --path-in="$sogma_xml_file" --path-out="$sogma_jsonl_file"
    rm -f "$sogma_xml_file"
fi

if [ ! -e $medbench_dir ]; then
    echo "$medbench_dir folder does not exist. Downloading and extracting..."
    mkdir $medbench_dir
    wget -nc "$medbench_url" -O "$medbench_zip_path"
    unzip "$medbench_zip_path" -d "$medbench_dir"
    rm -f "$medbench_zip_path"
    echo "Download and extraction complete."
fi
