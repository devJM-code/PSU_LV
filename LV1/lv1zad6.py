spamCount = 0
spamWordCount = 0
hamCount = 0
hamWordCount = 0
exclamCount = 0

with open("SMSSpamCollection.txt") as file:
for line in file:
line = line.strip()
if(line.startswith("spam")):
words = line.split(" ")
spamWordCount += len(words)
spamCount += 1
if(line[-1] == "!"):
exclamCount += 1
elif(line.startswith("ham")):
words = line.split(" ")
hamWordCount += len(words)
hamCount += 1

print(f"HamAvg: {hamWordCount / hamCount}\nSpamAvg: {spamWordCount / spamCount}\nSpam ending with !: {exclamCount}")
