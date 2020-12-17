echo "curl --header \"Content-Type: application/json\"   --request POST   --data \"{\"text\":\"${1}\"}\"   http://localhost:6543/predict_sentiment";
curl --header "Content-Type: application/json"   --request POST   --data "{\"text\":\"${1}\"}"   http://localhost:6543/predict_sentiment;
