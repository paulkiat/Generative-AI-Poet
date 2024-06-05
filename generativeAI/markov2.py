import random

input_text = "Tapped in on a different level trying to get to where I'm at, and I'm not ashamed"

with open('sonnets.txt') as f:
    input = f.read()
    
input = input.replace('\n', ' ')

words = input.split(" ")

# Second-order Markov Model
model = {}
for i in range(0, len(words) - 2):
    state1 = words[i]
    state2 = words[i+1]
    state3 = words[i+2]
    state = (state1, state2)
    if state not in model:
        model[state1] = []
    model[state1].append(state3)
# print(model)

# Generate text using second-order Markov Model
s = random.choice(list(model.keys()))
output = " ".join(s)
for i in range(10):
    if s not in model:
        break
    s = random.choice(model[s])
    output = output + " " + s
print(output)