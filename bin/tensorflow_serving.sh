GRN='\033[0;32m'
RED='\033[0;31m'
PURPLE='\033[0;34m'
RST='\033[0m'

if [ $# -ne 2 ]
  then
    echo "Usage ./tensorflow_serving.sh MODEL_PATH MODEL_NAME"
        exit 1
fi

MODEL_PATH=$1
MODEL_NAME=$2


echo "Serving Tensorflow model located at ${GRN}${MODEL_PATH}${RST} with name ${GRN}${MODEL_NAME}${RST}."
echo "Continue? (y/n)"
read prompt

if [ "$prompt" == "y" ]
    then
        docker run -d -t --rm -p 8501:8501 -v $MODEL_PATH/:/models/$MODEL_NAME -e MODEL_NAME=$MODEL_NAME tensorflow/serving
    else
        echo "Aborting..."
        exit 1
fi
