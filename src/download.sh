mkdir -p ./datasets/sts

function sts12() {
    wget http://ixa2.si.ehu.es/stswiki/images/4/40/STS2012-en-test.zip
    unzip STS2012-en-test.zip
    mv test-gold ./datasets/sts/sts12
    rm STS2012-en-test.zip
}

function sts13() {
    wget http://ixa2.si.ehu.es/stswiki/images/2/2f/STS2013-en-test.zip
    unzip STS2013-en-test.zip
    mv test-gs ./datasets/sts/sts13
    rm STS2013-en-test.zip
}

function sts14() {
    wget http://ixa2.si.ehu.es/stswiki/images/8/8c/STS2014-en-test.zip
    unzip STS2014-en-test.zip
    mv sts-en-test-gs-2014 ./datasets/sts/sts14
    rm STS2014-en-test.zip
}

function sts15() {
    wget http://ixa2.si.ehu.es/stswiki/images/d/da/STS2015-en-test.zip
    unzip STS2015-en-test.zip
    mv test_evaluation_task2a ./datasets/sts/sts15
    rm STS2015-en-test.zip
}

function sts16() {
    wget http://ixa2.si.ehu.es/stswiki/images/9/98/STS2016-en-test.zip
    unzip STS2016-en-test.zip
    mv sts2016-english-with-gs-v1.0 ./datasets/sts/sts16
    rm STS2016-en-test.zip
}

function stsb() {
    wget http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz
    tar -zxvf Stsbenchmark.tar.gz
    mv stsbenchmark ./datasets/sts/stsb
    rm Stsbenchmark.tar.gz
}

function sick() {
    wget http://alt.qcri.org/semeval2014/task1/data/uploads/sick_test_annotated.zip
    unzip sick_test_annotated.zip -d SICK
    mv SICK ./datasets/sts/sick
    rm sick_test_annotated.zip
}

function snli() {
    mkdir -p ./datasets/snli
    wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    unzip snli_1.0.zip -d snli
    mv snli ./datasets/snli/raw
    rm snli_1.0.zip
}

function mnli() {
    mkdir -p ./datasets/mnli
    wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
    unzip multinli_1.0.zip -d multinli
    mv multinli ./datasets/mnli/raw
    rm multinli_1.0.zip
}

# https://github.com/princeton-nlp/SimCSE/blob/30b08875a39d0e89d71f17c57bd0dcc18e7c2f15/data/download_nli.sh
function sup_simcse() {
    mkdir -p ./datasets/sup-simcse
    wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/nli_for_simcse.csv
    mv ./nli_for_simcse.csv ./datasets/sup-simcse/train.csv
}

sts12 >/dev/null 2>&1 &
sts13 >/dev/null 2>&1 &
sts14 >/dev/null 2>&1 &
sts15 >/dev/null 2>&1 &
sts16 >/dev/null 2>&1 &
stsb >/dev/null 2>&1 &
sick >/dev/null 2>&1 &

snli >/dev/null 2>&1 &
mnli >/dev/null 2>&1 &

sup_simcse >/dev/null 2>&1 &

wait
