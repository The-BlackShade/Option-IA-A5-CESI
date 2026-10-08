"""Complete WS_NLP_RNN_Etudiant (disaster tweets, GloVe, GRU and BiLSTM).

Fills every code blank, writes the theory answers, and fixes the 2020 code for
Keras 3 / NLTK. Markers become "# Solution" / "**Solution:**".
"""
from pathlib import Path
import re
import shutil
import nbformat

CESI = Path(r"C:\my\CESI\A5\Data Science")
PROJECT = Path(r"C:\my\Claude\cesi-data-science")
NAME = "WS_NLP_RNN_Etudiant_EN_relu.ipynb"
MARKER = re.compile(r"<em>\s*(?:PLEASE COMPLETE|TO COMPLETE|TO BE COMPLETED)\s*</em>", re.I)

CODE = {
6: r'''import os

# Solution: resolve the folder instead of os.chdir, which breaks when the cell
# is run twice (the second call looks for the folder inside itself).
from pathlib import Path

CANDIDATES = [
    Path('nlp-getting-started'),
    Path('.'),
    Path(r"C:\my\CESI\A5\Data Science\Data\nlp-getting-started"),
]
DATA = next((p for p in CANDIDATES if (p / 'train.csv').exists()), None)
if DATA is None:
    raise FileNotFoundError("train.csv not found. Looked in: " +
                            ", ".join(str(p.resolve()) for p in CANDIDATES))
print('reading from', DATA.resolve())

train_data = pd.read_csv(DATA / 'train.csv', usecols=['text', 'target'],
                         dtype={'text': str, 'target': np.int64})

train_data.shape''',

8: '''# Solution: the test set has no target column - that is what Kaggle asks you to predict.
# id is kept because the submission file is indexed by it.
test_data = pd.read_csv(DATA / 'test.csv', usecols=['id', 'text'], dtype={'id': np.int64, 'text': str})

test_data.shape''',

16: r'''# Function for cleaning each document: nlp_pipeline
# Tweet= tweet corpus = document
# A RegEx, or regular expression, is a sequence of characters that forms a search pattern.
def nlp_pipeline(text):
    # Convert uppercase letters to lowercase
    text = text.lower()

    # Replace new line with a space
    text = text.replace('\n', ' ').replace('\r', '')
    text = ' '.join(text.split())

    # Remove uppercase letters, all strings that are not letters or numbers
    # Solution: keep only letters, digits and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove special characters
    text = re.sub(r"(\s\-|-$)", "", text)
    # Solution: the dataset contains mojibake such as \x89Û_ ; drop any non-ASCII
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\x89û", "", text)
    # Solution: collapse the spaces left behind by the substitutions
    return ' '.join(text.split())''',

20: '''# Solution: NLTK ships a stopword list per language
mots_vides = stopwords.words('english')
print('\\n')
print(mots_vides)''',

21: '''# Function that removes stopwords: words that are very common in the language studied but don't make sense.
# as in French, the words: et, a, le, la, etc... (https://pythonspot.com/nltk-stop-words/ )
mots_vides=stopwords.words('english')

def remove_stopwords(sentence):
    words = sentence.split()
    # Solution
    words = [w for w in words if w not in mots_vides]

    return ' '.join(words)''',

23: '''# Solution: the standard punctuation characters
punctuations = string.punctuation
print(punctuations)''',

25: '''# Remove punctuation
def remove_punctuation(sentence):
    # Solution: str.maketrans builds a delete-table, translate applies it
    table = str.maketrans('', '', punctuations)

    words = [w.translate(table) for w in sentence.split()]

    return ' '.join(words)''',

27: '''# Remove emojis
def remove_emoji(sentence):
    emoji_pattern = re.compile("["
                                # emoticons
                           u"\\U0001F600-\\U0001F64F"
                               # symbols & pictographs
                           u"\\U0001F300-\\U0001F5FF"
                                # transport & map symbols
                           u"\\U0001F680-\\U0001F6FF"
                               # flags (iOS)
                           u"\\U0001F1E0-\\U0001F1FF"
                           u"\\U00002702-\\U000027B0"
                           u"\\U000024C2-\\U0001F251"
                           "+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', sentence)''',

29: '''from nltk.stem import WordNetLemmatizer
lem=WordNetLemmatizer()

def lem_word(sentence):
    words = sentence.split()
    # Solution: lemmatise brings a word back to its dictionary form (better -> good),
    # unlike the stemmer above which just chops the ending (emergency -> emerg).
    words = [lem.lemmatize(w) for w in words]
    return ' '.join(words)''',

31: '''# Remove words with less than two characters
def remove_small(sentence):

    words = sentence.split()
    # Solution: very short tokens are mostly leftovers of the cleaning
    words = [w for w in words if len(w) > 2]
    return ' '.join(words)''',

33: '''def clean_text(data):
    data['text'] = data['text'].apply(lambda x : remove_url(x))
    data['text'] = data['text'].apply(lambda x : remove_html(x))
    #data['text'] = data['text'].apply(lambda x : stem_words(x))
    data['text'] = data['text'].apply(lambda x : remove_punctuation(x))
    data['text'] = data['text'].apply(lambda x : remove_stopwords(x))
    data['text'] = data['text'].apply(lambda x : remove_emoji(x))
    data['text'] = data['text'].apply(lambda x : remove_small(x))
    data['text'] = data['text'].apply(lambda x : lem_word(x))
    data['text'] = data['text'].apply(lambda x : nlp_pipeline(x))
    return data''',

37: '''# Solution: count the tweets the cleaning emptied, then drop them
print('empty tweets after cleaning:', (D.text.str.strip() == '').sum())

# Eliminate empty tweets if any.
D = D[D.text.str.strip() != ''].reset_index(drop=True)

print(D.shape)

D.text.head()''',

39: '''# You need to install wordCloud
#!pip install wordcloud

from wordcloud import WordCloud

# Create Wordcloud from tweets
# https://www.geeksforgeeks.org/generating-word-cloud-python/
D.text
all_words = ' '.join([text for text in D.text])
# Solution
wordcloud = WordCloud(width=800, height=500, random_state=21,
                      max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()''',

41: '''# Learning-testing partition
from sklearn.model_selection import train_test_split
# Solution: stratify keeps the same share of disaster tweets in both parts
dtrain, dtest = train_test_split(D, test_size=0.2, random_state=42, stratify=D['target'])

# Checking the split
print(dtrain.shape)
print(dtest.shape)''',

43: '''# Tokenization with Keras
# Solution: "from keras.preprocessing.text import Tokenizer" no longer exists in
# Keras 3; the tf.keras path below still works.

def define_tokenizer(train_sentences, val_sentences, test_sentences):
    sentences = pd.concat([train_sentences, val_sentences, test_sentences])
    tokenizer = tf.keras.preprocessing.text.Tokenizer()

    ## Creation of the dictionary from the sample documents
    tokenizer.fit_on_texts(sentences)

    return tokenizer

def encode(sentences, tokenizer):
    encoded_sentences = tokenizer.texts_to_sequences(sentences)
    # Solution: tweets have different lengths; pad with zeros so the result is a
    # rectangular array, which is what from_tensor_slices needs.
    encoded_sentences = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_sentences, padding='post')

    return encoded_sentences''',

45: '''# Solution: the dictionary is built on every split, so no word is unknown at encoding time.
# (Note for the report: fitting on test text too is acceptable here because the
# tokenizer only assigns ids - no label is involved - but it would be cleaner to
# fit on the training part alone.)
tokenizer = define_tokenizer(dtrain['text'], dtest['text'], test_data_c['text'])

encoded_sentences = encode(dtrain['text'], tokenizer)
val_encoded_sentences = encode(dtest['text'], tokenizer)
encoded_test_sentences = encode(test_data_c['text'], tokenizer)

# Number of documents processed
print(tokenizer.document_count)''',

50: '''from zipfile import ZipFile
from pathlib import Path

# Solution: extract only if it has not been done yet, and say clearly what is
# missing otherwise. glove.6B.zip is ~822 MB from https://nlp.stanford.edu/data/
GLOVE_TXT = Path('glove.6B/glove.6B.100d.txt')

if not GLOVE_TXT.exists():
    if Path('glove.6B.zip').exists():
        with ZipFile('glove.6B.zip', 'r') as zf:
            zf.extractall('glove.6B')
    else:
        raise FileNotFoundError(
            "glove.6B.zip not found. Either download it from "
            "https://nlp.stanford.edu/data/glove.6B.zip, or install gensim and run:\\n"
            "  import gensim.downloader as api\\n"
            "  kv = api.load('glove-wiki-gigaword-100')   # ~130 MB, same vectors\\n"
            "  kv.save_word2vec_format('glove.6B/glove.6B.100d.txt', write_header=False)")

print(GLOVE_TXT, 'ready')''',

52: '''embedding_dict = {}

f = open(GLOVE_TXT, 'r', encoding='utf-8')
for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:],'float32')
        # Transform each word into a vector of dimension 100.
        # Solution
        embedding_dict[word] = vectors

f.close()


# Check number of terms
print('Found %s word vectors.' % len(embedding_dict))''',

56: '''# Initial emmbedding matrix for our dataset
hit=0
misses=[]

# Number of tokens ( numpy is zero-based)
# Solution: +1 because Keras reserves index 0 for padding
num_words = len(tokenizer.word_index) + 1

# Dimension of representation =100 according to Glove chosen
embedding_matrix = np.zeros((num_words, 100))

# Fill the matrix with the coordinates from the pre-trained representation
# Provided that the dictionary term we're looking for is present in GloVe's pre-trained representation
for word, i in tokenizer.word_index.items():
    if i > num_words:
        continue

    emb_vec = embedding_dict.get(word)

    if emb_vec is not None:
        embedding_matrix[i] = emb_vec
        hit = hit+1
    else:
        misses.append(word)

# Control display: the number of terms found and not found in GloVe's pre-trained representation.
# Solution
print('Converted %d words (%d misses)' % (hit, len(misses)))''',

63: '''def pipeline(tf_data, buffer_size=100, batch_size=32):
    tf_data = tf_data.shuffle(buffer_size)
    tf_data = tf_data.prefetch(tf.data.AUTOTUNE)

    # Solution: padded_batch pads each batch to its own longest sentence.
    # padded_shapes: [None] for the variable-length sentence, [] for the scalar label.
    tf_data = tf_data.padded_batch(batch_size, padded_shapes=([None], []))

    return tf_data

tf_data = pipeline(tf_data, buffer_size=1000, batch_size=32)

print(tf_data)''',

65: '''# Solution: the validation set, built the same way but never shuffled
tf_val_data = tf.data.Dataset.from_tensor_slices((val_encoded_sentences, dtest['target'].values))

def val_pipeline(tf_data, batch_size=1):
    tf_data = tf_data.prefetch(tf.data.AUTOTUNE)
    tf_data = tf_data.padded_batch(batch_size, padded_shapes=([None], []))

    return tf_data

tf_val_data = val_pipeline(tf_val_data, batch_size=len(dtest))

print(tf_val_data)''',

73: '''# Creation of model 1 architecture
model1 = tf.keras.Sequential([
    embedding,
    tf.keras.layers.SpatialDropout1D(0.2),
    # GRU with 128 neurons and dropout=0.2
    # Solution
    tf.keras.layers.GRU(128, dropout=0.2)
    ,tf.keras.layers.Dense(1, activation='sigmoid')
])

model1.summary()''',

75: '''model1.compile(
    # Solution: one sigmoid output, two classes -> binary cross-entropy
    loss='binary_crossentropy'
    , optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=['accuracy']
)''',

82: '''model2 = tf.keras.Sequential([
    embedding,
    tf.keras.layers.SpatialDropout1D(0.4),
    # bi-directional single hidden layer LSTM with 128 neurons and dropout equal to 0.3
    # Solution: Bidirectional runs one LSTM forward and one backward, then
    # concatenates them, so each word is read with its left and right context.
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.3)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model2.compile(
    # loss function binary cross entropy
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.01),
    # accuracy metric
    metrics=['accuracy']
)

model2.summary()

history2 = model2.fit( tf_data, validation_data = tf_val_data, epochs = 50 )

fig, axs = plt.subplots(1, 2, figsize=(20, 5))

axs[0].set_title('Loss')
axs[0].plot(history2.history['loss'], label='train')
axs[0].plot(history2.history['val_loss'], label='val')
axs[0].legend()

axs[1].set_title('Accuracy')
# Accuracy representation for axis 2 data train
axs[1].plot(history2.history['accuracy'], label='train')
# Accuracy representation for data val in axis 2
axs[1].plot(history2.history['val_accuracy'], label='val')
axs[1].legend()''',

87: '''# GRU model evaluation on the test part
# Solution: evaluate takes the DATA, not the predictions. The original line
# (model1.evaluate(dtest['prediction1'], dtest['target'])) feeds the predictions
# back in as if they were tweets, which is meaningless.
score1 = model1.evaluate(tf_val_data, verbose=1)

print("Validation Score model1:", score1[0])
print("Validation Accuracy model1:", score1[1])''',

89: '''# Evaluation of the LSTM model on the test part
# Solution
score2 = model2.evaluate(tf_val_data, verbose=1)

# Display error: Validation Score model1 and accuracy of model 2: Validation Accuracy model2
print("Validation Score model2:", score2[0])
print("Validation Accuracy model2:", score2[1])''',

92: '''# Solution: predicted disaster (1) while the truth was not a disaster (0)
false_positives = dtest[(dtest['prediction2'] == 1) & (dtest['target'] == 0)]

print('Count of false positives: ' + str(len(false_positives)))

false_positives.head(10)''',

94: '''# Solution: predicted not a disaster (0) while it really was one (1)
false_negatives = dtest[(dtest['prediction2'] == 0) & (dtest['target'] == 1)]

print('Count of false negatives: ' + str(len(false_negatives)))

false_negatives.head(10)''',

98: '''# Solution: Kaggle expects two columns, id and target
submission = pd.DataFrame({'target': predictions})
submission.index = test_data['id']
submission.index.name = 'id'
submission.to_csv('submission.csv')


submission.head()''',

100: '''def compare_words(train_words, test_words):
    unique_words = len(np.union1d(train_words, test_words))
    # Solution: words present in both sets
    matching = len(np.intersect1d(train_words, test_words))
    not_in_train = len(np.setdiff1d(test_words, train_words))
    not_in_test = len(np.setdiff1d(train_words, test_words))

    print('Number of words in both parts dtrain and dtest: ' + str(unique_words))
    print('Number of words in matching: ' + str(matching))
    print('Number of words in the dtrain dataset and not in the test dataset: ' + str(not_in_test))
    print('Number of words in the test dataset and not in the dtrain dataset: ' + str(not_in_train))

# Comparison between training data and validation data
compare_words(encoded_sentences, val_encoded_sentences)''',
}

ANSWERS = {
69: ['''**Solution:** the vanishing gradient is what stops a simple recurrent network from learning long dependencies.

During backpropagation through time, the gradient is multiplied by the same recurrent weight matrix at every step. If those values are below 1, the product shrinks exponentially: after twenty words the signal coming back to the first word is practically zero. The weights that would capture a long-range dependency therefore never move.

In practice the network keeps only a short memory: it learns the link between neighbouring words and forgets the beginning of the sentence. For these tweets it would miss a negation placed early ("this is not a fire, just a sunset"), because by the time it reads the last word the first one has faded.

The mirror problem exists too: with values above 1 the product explodes, the loss becomes NaN, and training dies. That one is handled by gradient clipping.'''],

71: ['''**Solution:** by giving the network a memory path that is **added to** rather than multiplied through, and by letting it learn what to keep.

That is what gating does. Each step computes sigmoid gates - values between 0 and 1 - that decide how much of the state to forget, how much of the new word to write, and how much to expose. The state update becomes `c = f * c_prev + i * g`: when the forget gate stays near 1, both the information and the gradient travel many steps almost unchanged.

Other complementary measures: gradient clipping against explosion, shorter sequences, and a pre-trained embedding (as here with GloVe) so the model has less to learn from few examples.''',

'''**Solution:** among the three architectures seen, the **LSTM** and the **GRU** solve it; the **simple RNN** (`SimpleRNN`) does not.

- **LSTM**: three gates (forget, input, output) plus a separate cell state used as long-term memory.
- **GRU**: two gates (update, reset), no separate cell state - simpler and about 25% cheaper, usually as accurate.
- **SimpleRNN**: overwrites its whole state at each step, so it suffers the full vanishing gradient.

This workshop builds a GRU (model 1) and a bidirectional LSTM (model 2), which is why their scores can be compared.'''],
}


path = CESI / NAME
src = PROJECT / NAME
if not path.exists():
    shutil.copy2(src, path)

nb = nbformat.read(path, as_version=4)
for idx, source in CODE.items():
    assert nb.cells[idx].cell_type == "code", f"cell {idx} is not code"
    nb.cells[idx].source = source
    nb.cells[idx].outputs, nb.cells[idx].execution_count = [], None
for idx, texts in ANSWERS.items():
    cell = nb.cells[idx]
    assert cell.cell_type == "markdown", f"cell {idx} is not markdown"
    found = MARKER.findall(cell.source)
    assert len(found) == len(texts), f"cell {idx}: {len(found)} markers, {len(texts)} answers"
    for text in texts:
        cell.source = MARKER.sub(lambda _m: text, cell.source, count=1)

backup = path.with_suffix(".ipynb.bak")
if not backup.exists():
    shutil.copy2(src, backup)
nbformat.write(nb, path)
shutil.copy2(path, src)

code_left = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and "PLEASE COMPLETE" in c.source]
md_left = [i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and MARKER.search(c.source)]
print(f"{NAME}: {len(CODE)} code cells, {sum(len(v) for v in ANSWERS.values())} answers written")
print("code blanks left:", code_left, "| markdown markers left:", md_left)
