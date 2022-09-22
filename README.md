# Base Derivative Maximum Entropy Phonological Learner
This is an initial implementation of a Maximum Entropy learner with base-derivative constraints ().

## 1. Setting up the data
The data is set up as a CSV consisting of 4 parts:
    1. The segmented underlying form for the word
    2. An index corresponding to the winning candidate
    3. The lexemes of the words
    4. The segment candidates of the word

### 1.1 The underlying form
The underlying form is a string comprised of one or more morphemes. If there is more than one morpheme, the boundary between them is denoted by the hyphen (```-```), as shown below:
    * ```bat``` consists of a single morpheme: /bat/
    * ```bat-a``` consists of two morphemes: /bat/ and /a/

Make sure that the lexemes of the word and the number of morphemes are the same (to be elaborated on later).

### 1.2 The winning candidate
The index simply is an integer that points to the winning candidate among the specified candidates for that word.

### 1.3 The lexemes
Each word consists of one or more morphemes, each in turn corresponding to a lexeme. Different lexemes are delimited using the semicolon (```;```), as shown below:
    * ```1;A``` corresponds to the lexemes labelled ```1``` and ```A```

The order of the morphemes after segmenting of the underlying form and candidate are identical. For example, consider the example below:
    * Given ```bat-a``` and ```1;A```, ```1``` corresponds to ```bat``` and ```A``` corresponds to ```a```

### 1.4 The candidates
Each underlying form is associated with a finite set of candidates. Each candidate it delimited using the semicolon (```;```), as shown below:
    * ```bat;bac``` corresponds to the candidates [bat] and [bac]

## 2. Setting up the constraints
Constraints are written along the columns instead of the rows, like above. This is mainly so the code has an easier time reading in the relevant information. The constraints file is set up as a CSV consisting of 3 parts:
    1. The name of the constraint
    2. The type of the constraint
    3. The constraint definition

### 2.1 The name of the constraint
As the name suggests, it is simply a readable label for the constraint you are defining.

### 2.2 The type of the constraint
This code supports 5 different kinds of constraints:
    1. Markedness constraint ```M```
    2. Insertion faithfulness constraints ```I```
    3. Deletion faithfulness constraints ```D```
    4. Substitution faithfulness constraints ```S```
    2. Insertion base-derivative constraints ```BI```
    3. Deletion base-derivative constraints ```BD```
    4. Substitution base-derivative constraints ```BS```

### 2.3 The constraint definitions
All constraints are defined using regular expressions over strings. A constraint is specifically defined differently based on its type.
    1. For markedness constraints, it is just a string corresponding to a regular expression (ex. ```ti```)
    2. For faithfulness constraints, it is a tuple corresponding to definitions on the input and output. A tuple is delimited in the ```.csv``` using a semicolon, as above (ex. ```[tk];c``` says that the input cannot be either [t] or [k] where its corresponding segment is [c]). Note that since general deletion and insertion processes do not look at specific segments, you do not need to define anything (ex. ```;```)
    3. For base-derivative constraints, it is identical to the faithfulness constraints, except they can have an optional first parameter corresponding to a context-specific requirement in the base in order for the constraint to apply. To specify the optional parameter, simply include an additional semicolon to the definition (```pos;base;deriv```)

## 3. Running the data
In order to run the code, you must specify a dataset and a constraint set. Optionally, you can also specify the number of iterations to run the model for. To run some sample code, use the following command:
```
python3 learn.py --data data/feeding.csv --constraints data/constraints.csv --iterations 1000
```

## 4. Notes
This code is still being worked on and is missing a few key components. Thee main one of concern is how the model determines its base. The code determines a base by slicing off a morpheme from the right edge and checking to see if this word exists elsewhere in the dataset (e.g. the potential base of [bat-a] is [bat]). This has several issues. First, it does not allow for checking whether there are prefixes or infixes in the word, and rather assumes the left-most morpheme is the stem. Second, the model does not actually check whether the suffix being removed is a derivational suffix. This notion of base-derivative correspondence is typically associated with derivational morphology, and thus should in principle ignore inflectional suffixes. These issues will be addressed in the near future.

## References
<a id="1">[1]</a> McCarthy, J. J., & Prince, A. (1995). Faithfulness and reduplicative identity. *Linguistics Department Faculty Publication Series*, 10.
