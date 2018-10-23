valgroup="/aws/lambda/ss_validator"
echo "checking for log streams in the "$valgroup" log group" 


# Delete the current log streams so we can separate fresh logs from previous version logs
validator=$(aws logs describe-log-streams --log-group-name $valgroup)
streams=`jq -n "$validator" | jq '.[] |.[] | .logStreamName'`
for i in $streams; do	
	temp="${i%\"}"  	#remove trailing quote
	temp="${temp#\"}"	#remove leading quote
	echo "deleting "$i""
	aws logs delete-log-stream --log-group-name $valgroup --log-stream-name $temp	
done

processgroup="/aws/lambda/ss_process"
echo "checking for log streams in the "$processgroup" log group"

process=$(aws logs describe-log-streams --log-group-name $processgroup)
streams=`jq -n "$process" | jq '.[] |.[] | .logStreamName'`
for i in $streams; do   
        temp="${i%\"}"          #remove trailing quote
        temp="${temp#\"}"       #remove leading quote
        echo "deleting "$i""
        aws logs delete-log-stream --log-group-name $processgroup --log-stream-name $temp   
done

