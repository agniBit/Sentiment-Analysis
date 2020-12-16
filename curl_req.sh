echo "curl --header \"Content-Type: application/json\"   --request GET   --data \"{\"text\":\"${1}\"}\"   http://localhost:6543/get_data";
curl --header "Content-Type: application/json"   --request GET   --data "{\"text\":\"${1}\"}"   http://localhost:6543/get_data;
