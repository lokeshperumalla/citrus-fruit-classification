# Citrus fruit classification: two approaches

Compares (A) hand-crafted colour/texture/lesion features with SVM and
Random Forest against (B) a small CNN trained from scratch, on a four-class
orange dataset (fresh, grenning, canker, blackspot).

Report: Citrus_CV_Report.pdf

## Run
pip install torch torchvision scikit-learn scikit-image opencv-python-headless imagehash matplotlib pandas joblib scipy
Edit the ROOT/OUT paths at the top of each script, then run in order:
eda.py, prep.py, classical.py, cnn.py, aggregate.py, predict.py
