python setup/macos/install_env.py
pip install --upgrade pip
echo "Activating 'debiased' virtual environment..."
source debiased/bin/activate
pip install -r setup/requirements.txt
echo "Do you want to use your own dataset? (yes or no)"
read own_dataset
while [[ "$own_dataset" == "yes" ]]; do
    if [[ "$own_dataset" == "yes" ]]; then
        echo "We are ready to load your own dataset!..."
        echo "First, tell us where your data is located (give us the absolute path)"
        read own_path
        echo "Now, rename the columns of your excel in the following way:"
        echo "All features should start with 'feature ' For example, if one feature is 'previous grade'n call if 'feature previous grade'"
        echo "All demographic attributes you want to mitigate on should start with 'demo'"
        echo "The label on which to do the classification should be called 'label'"
        echo "When you are ready, type the name of your dataset"
        read dataset_name
        python setup/macos/convert_data.py --path $own_path --name $dataset_name

        echo "Do you want to load another dataset? (yes or no)"
        read own_dataset
    fi
done
echo "Data Import Done!"
python setup/macos/run_fairness_gui.py




