
for n in {1..1}
do
    for i in {1..50}
    do
        # python -m imitation.scripts.train_adversarial with airl sorting_onions$n session_number=$n
        python -m imitation.scripts.train_adversarial with airl sorting_onions512 session_number=512
        echo ""
    done
done

echo "Message Body" | mail -s "experiment finished" sarora@udel.edu
