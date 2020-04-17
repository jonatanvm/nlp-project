'''
Process plenary transcripts from 2008-2016 into two speech-level csv files.

Detect and remove speeches in Swedish
Replace any special whitespace chars with " "

output: 
two csv files with columns [speaker, date, speech]: speeches-1.csv, speeches-2.csv
delimiter = "|", line terminator = "\n" 
output written to current directory

'''
import csv
import os
import re

import pandas as pd
from langdetect import detect

folder = input("insert path to transcript main folder ")


##############################################################

def getFilePaths(path, suffix=".transcript"):
    '''
    Returns paths to files in subfolders with suffix
    
    input: path to main folder, optional: file suffix
    output: a list of filepaths
    '''
    filelist = []

    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith((suffix)):
                filelist.append(root + "/" + name)
    return filelist


def cleanSpeaker(speakerline):
    '''
    remove party, speech type, title from speaker name

    input: string
    output: string
    '''
    chairpattern = re.compile(r"^.*uhemies")
    speakerline = re.sub(chairpattern, "", speakerline)
    # if speakerline.find("uhemies") != -1:
    #    speakerline = speakerline.split("uhemies ")[1]

    if speakerline.find("inisteri") != -1:
        speakerline = speakerline.split("inisteri ")[1]

    if speakerline.find("/") != -1:
        speakerline = speakerline.split("/")[0]

    speaker = speakerline.strip()

    return speaker


def cleanText(text):
    '''
    Remove line breaks and replace all whitespace chars with space

    input: string
    output: string
    '''
    stripwords = ["PMPVUORO", "KESKUST", "KYSKESK", "ASIAKOHTA", "EDPVUORO"]

    text = text.replace(u'\xa0', u' ')
    text = text.replace('\n', ' ')
    words = text.split()
    words = [w for w in words if w not in stripwords]
    text = " ".join(words)
    return text


filelist = getFilePaths(folder)

with open('speeches.csv', 'w') as outfile:
    writer = csv.writer(outfile, lineterminator='\n', delimiter='|')
    writer.writerows([["speaker", "date", "speech"]])

for file in filelist:
    date = file.split('/')[-2]
    print("File: ", file, "Date: ", date)

    # read transcript
    transcript = open(file, "r", encoding="UTF-8").read()

    speeches = transcript.split("SPEAKER: ")

    for speech in speeches:
        # Extract from the beginning of speech until first colon or linebreak
        firstcolon = re.compile(r"^([^:\n]+?)[:\n]")
        speaker = re.findall(firstcolon, speech)
        speech = re.sub(firstcolon, "", speech)

        if speaker:
            speaker = cleanText(speaker[0])
            speaker = cleanSpeaker(speaker)

        if not speaker:
            speaker = ""

        lines = speech.split("\n")
        finnish = ""

        for l in lines:
            # Detect language
            try:
                lan = detect(l)
            except:
                lan = ""

            if lan != "sv":
                finnish += (" " + l)

        finnish = cleanText(finnish)
        if finnish.strip() != "":
            with open('speeches.csv', 'a') as outfile:
                writer = csv.writer(outfile, lineterminator='\n', delimiter='|')
                writer.writerows([[speaker, date, finnish]])

    print("\n Finished with file %s" % (file))

# Read full data back in to split it in roughly half
df = pd.read_csv("speeches.csv", delimiter="|", lineterminator="\n")

# Split df
df1 = df.iloc[:50000]
df2 = df.iloc[50000:]

print(len(df1))
print(len(df2))

df1.to_csv("speeches-1.csv", sep="|", line_terminator="\n")
df2.to_csv("speeches-2.csv", sep="|", line_terminator="\n")
