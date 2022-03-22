import re
import numpy as np


def clean_entity(element):
    element = re.sub(r"[^a-zA-Z0-9]+", ' ', element)
    element = element.lower()
    return element


class VerbalisationTriplet:
    def __init__(self, triplet, pos_tagger, tokenizer):
        self.indexed_subject = triplet['subject'][0]
        self.indexed_object = triplet['object'][0]
        self.pos_tagger = pos_tagger
        self.tokenizer = tokenizer
        self.set_clean_triplet_elements(triplet)
        self.verblike_pos_tags = ('VERB', 'AUX')
    
    def __getitem__(self, arg):
        return getattr(self, arg)
    
    def set_clean_triplet_elements(self, triplet):
        self.subject = clean_entity(triplet['subject'][1])
        self.property = clean_entity(triplet['property'])
        self.object = clean_entity(triplet['object'][1])
    
    def contains_verb(self, predicate):
        return self.verblike_pos_tags in {pos_tag['entity'] for pos_tag in self.pos_tagger(predicate)}
    
    def get_masked_statement(self, negate=False, prefix_entities=True):
        #masked sentence in form [SOS, subject prefix, subject, predicate prefix, predicate, object prefix, object, EOS]
        masked_elements = list()
        token_ranges = {
            'subject': {},
            'property': {},
            'object': {},
            }
        masked_elements.append('<s>')
        if prefix_entities:
            masked_elements.append(' <mask>')
        else:
            masked_elements.append([])
        masked_elements.append(f" {self.subject}")
            
        # if negate:
        #     masked_elements.append(self.prefix_mask('not'))
        if not self.contains_verb(clean_entity(f" {self.property}")):
            masked_elements.append(' <mask>')
        else:
            masked_elements.append([])
        masked_elements.append(f" {self.property}")
            
        if prefix_entities:
            masked_elements.append(' <mask>')
        else:
            masked_elements.append([])
        masked_elements.append(f" {self.object}")
        masked_elements.append('.</s>')
        
        return masked_elements
    
    def get_string_verbalization(self, negate=False):
        masked_elements = list()
        masked_elements.append(self.subject)
        if negate:
            masked_elements.append('not')
        masked_elements.append(self.property)
        masked_elements.append(self.object)
        masked_statement = ' '.join(masked_elements)
        masked_statement += '.</s>'
        return masked_statement

    @staticmethod
    def prefix_mask(element: str) -> str:
        return f"<mask> {element}"
    
    def mask_individuals(self, sentence):
        num_subject_tokens = self.tokenizer.encode(sentence[2], return_tensors="pt", add_special_tokens=False).shape[1]
        num_object_tokens = self.tokenizer.encode(sentence[6], return_tensors="pt", add_special_tokens=False).shape[1]
        masked_sentence = sentence.copy()
        masked_sentence[2] = ' '.join(['<mask>']*num_subject_tokens)
        masked_sentence[6] = ' '.join(['<mask>']*num_object_tokens)
        return masked_sentence

    def mask_predicate(self, sentence):
        predicate_length = len(self.tokenizer(self.property)['input_ids']) - 2
        masks = ' '.join(['<mask>'] * predicate_length)
        sentence = sentence.replace(self.property, masks)
        return sentence

    def mask_predicates(self, sentences):
        return [self.mask_predicate(sentence) for sentence in sentences]

    @staticmethod
    def replace_last(sentence, old, new):
        return (sentence[::-1].replace(old[::-1],new[::-1], 1))[::-1]    

    
class TripletSet:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
    
    def verbalize_triplet(self, masked_statement):
        masked_statement_ids = self.tokenizer(masked_statement, return_tensors='pt')
        generated_ids = self.model.generate(
            masked_statement_ids['input_ids'], 
            output_scores=True,
            return_dict_in_generate=True,
            num_beams=6, 
            num_return_sequences=3,
            max_length=masked_statement_ids['input_ids'].shape[1] + masked_statement.count('<mask>')*2, 
            min_length=0, 
            length_penalty=0.1,
            num_beam_groups=2,
            diversity_penalty=0.1,
            early_stopping=True,
            repetition_penalty=5.,
        )
        sentences = self.tokenizer.batch_decode(generated_ids['sequences'], skip_special_tokens=True)
        sentences_clipped = [sentence.split('.')[0] + '.' for sentence in sentences]
        scores = np.exp(generated_ids['sequences_scores']).tolist()
        return sentences, scores
    
    
    
def replace_triplet_entities(triplet, replacement_dict):
    new_triplet = dict()
    new_triplet['subject'] = replacement_dict[triplet['subject'][0]]
    new_triplet['property'] = clean_entity(triplet['property'])
    new_triplet['object'] = replacement_dict[triplet['object'][0]]
    return new_triplet

def find_entity_phrases(sentence, subject, predicate, _object):
    cleaned_sentence = clean_entity(sentence.replace('<s>', ''))
    
    predicate_split_sentence_pieces = cleaned_sentence.split(predicate)
    if len(predicate_split_sentence_pieces) != 2:
        return None

    subject_split_sentence_pieces = predicate_split_sentence_pieces[0].split(subject)
    object_split_sentence_pieces = predicate_split_sentence_pieces[1].split(_object)
    if len(subject_split_sentence_pieces) != 2 or len(object_split_sentence_pieces) != 2:
        return None

    subject_prefix = subject_split_sentence_pieces[0]
    predicate_prefix = subject_split_sentence_pieces[1]
    object_prefix =object_split_sentence_pieces[0]

    subject_phrase = f'{subject_prefix} {subject}'.strip()
    predicate_phrase = f'{predicate_prefix} {predicate}'.strip()
    object_phrase =f'{object_prefix} {_object}'.strip()

    return subject_phrase, predicate_phrase, object_phrase
    
    
    
class PremiseSet(TripletSet):
    def __init__(self, triplets):
        super().__init__(triplets)
        

        
class HypothesisSet(TripletSet):
    def __init__(self, triplets):
        super().__init__(triplets)