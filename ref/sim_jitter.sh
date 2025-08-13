tc qdisc del dev eth1 root
tc qdisc add dev eth1 root handle 1: htb
tc class add dev eth1 parent 1: classid 1:1 htb rate 1000Mbit
tc qdisc add dev eth1 parent 1:1 netem delay 0ms loss 0% limit 500
tc filter add dev eth1 parent 1: prio 4 protocol ip u32 match ip dst ${1} flowid 1:1
tc qdisc show dev eth1
tc -s qdisc show dev eth1

while true
do
    delay=$(( $RANDOM % ${2} + 1 ))
    echo "$(date +"%Y-%m-%d %T.%3N") add random delay ${delay} ms"

    tc qdisc change dev eth1 parent 1:1 netem delay ${delay}ms loss 0% limit 500
    sleep 0.01
done