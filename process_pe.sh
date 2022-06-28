echo 'Load data'
cat data/acl2017-neural_end2end_am/data/conll/Essay_Level/train.dat \
    data/acl2017-neural_end2end_am/data/conll/Essay_Level/test.dat \
    data/acl2017-neural_end2end_am/data/conll/Essay_Level/dev.dat > \
    data/acl2017-neural_end2end_am/data/conll/Essay_Level/all.dat
cat data/acl2017-neural_end2end_am/data/conll/Paragraph_Level/train.dat \
    data/acl2017-neural_end2end_am/data/conll/Paragraph_Level/test.dat \
    data/acl2017-neural_end2end_am/data/conll/Paragraph_Level/dev.dat > \
    data/acl2017-neural_end2end_am/data/conll/Paragraph_Level/all.dat
python load_text_essays.py \
    --tag \
    --data-path data/acl2017-neural_end2end_am/data \
    > data/PE_token_level_data.tsv
python prepare_data.py \
    --data-path data/PE_token_level_data.tsv \
    --data-save-path data/PE_data_df.csv \
    --vocab-save-path data/bow_vocab.json

echo 'Add special tokens to bert vocab'
special_tokens=('<essay>' '<para-conclusion>' '<para-body>' '<para-intro>' '<ac>' \
                '<\/essay>' '<\/para-conclusion>' '<\/para-body>' '<\/para-intro>' '<\/ac>')
for i in {0..9}
do
sed -i "s/\[unused$i\]/${special_tokens[$i]}/" data/bert-base-uncased/vocab.txt
done
echo 'Done'


