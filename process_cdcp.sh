python preprocess_cdcp/marseille_preprocess_cdcp.py 

echo 'Add special tokens to bert vocab'
special_tokens=('<essay>' '<para-conclusion>' '<para-body>' '<para-intro>' '<ac>' \
                '<\/essay>' '<\/para-conclusion>' '<\/para-body>' '<\/para-intro>' '<\/ac>')
for i in {0..9}
do
sed -i "s/\[unused$i\]/${special_tokens[$i]}/" data/bert-base-uncased/vocab.txt
done
echo 'Done'

python bert_tokenize_for_cdcp.py
