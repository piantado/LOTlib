
# Run the standard unit tests

echo "--------------- Starting tests on LOTlib at" $(date) " ---------------"

python -m unittest discover

echo "--------------- Ending tests on LOTlib at" $(date) " ---------------"