# sourced from Let's Stop Incorrect Comparisons in End-to-End Relation Extraction!
# https://raw.githubusercontent.com/btaille/sincere/dd1c34916ddcdc5ceb2799d64b17e80cdf1a5b31/ace_preprocessing/ace2005/manual_fix.py
# Manual fix tokenization issue in doc "CNN_IP_20030408.1600.04"
# "JEFF GREENFIELD , CNN SR. ANALYST ( voice-over )" becomes "JEFF GREENFIELD , CNN SR. . ANALYST ( voice-over )"

with open("text/CNN_IP_20030408.1600.04.split.txt", "r", encoding="utf8") as file:
    doc = [line.replace("SR. .", "SR.") for line in file]

with open("text/CNN_IP_20030408.1600.04.split.txt", "w", encoding="utf8") as file:
    file.write("\n".join(doc))
