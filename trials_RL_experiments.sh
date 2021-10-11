
for i in {1..1}
do
    python -m imitation.scripts.expert_demos
    echo ""
done

echo "Message Body" | mail -s "RL experiment finished" sarora@udel.edu
