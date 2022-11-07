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
        self.mask_token = tokenizer.mask_token
        self.set_clean_triplet_elements(triplet)
        self.verblike_pos_tags = {'VERB', 'AUX'}
    
    def __getitem__(self, arg):
        return getattr(self, arg)
    
    def set_clean_triplet_elements(self, triplet):
        self.subject = clean_entity(triplet['subject'][1])
        self.property = clean_entity(triplet['property'])
        self.object = clean_entity(triplet['object'][1])
    
    def contains_verb(self, predicate):
        pos_tags = {pos_tag['entity'] for pos_tag in self.pos_tagger(predicate)}
        return len(pos_tags.intersection(self.verblike_pos_tags)) > 0
    
    def get_masked_statement(self, mask_token=None, negate=False, prefix_entities=True):
        if mask_token is None:
            mask_token = self.mask_token
        #masked sentence in form [SOS, subject prefix, subject, predicate prefix, predicate, object prefix, object, EOS]
        masked_elements = list()
        masked_elements.append('<s>')
        if prefix_entities:
            masked_elements.append(mask_token)
        else:
            masked_elements.append('')
        masked_elements.append(self.subject)
            
        # if negate:
        #     masked_elements.append(self.prefix_mask('not'))
        if not self.contains_verb(clean_entity(self.property)):
            masked_elements.append(mask_token)
        else:
            masked_elements.append('')
        masked_elements.append(self.property)
            
        if prefix_entities:
            masked_elements.append(mask_token)
        else:
            masked_elements.append('')
        masked_elements.append(self.object)
        masked_elements.append('.</s>')
        
        return masked_elements
    
    def mask_individuals(self, sentence, use_single_mask=True):
        masked_sentence = sentence.copy()
        if use_single_mask:
            masked_sentence[2] = self.mask_token
            masked_sentence[6] = self.mask_token
        else:
            num_subject_tokens = self.tokenizer.encode(sentence[2], return_tensors="pt", add_special_tokens=False).shape[1]
            num_object_tokens = self.tokenizer.encode(sentence[6], return_tensors="pt", add_special_tokens=False).shape[1]
            masked_sentence[2] = ' '.join([self.mask_token]*num_subject_tokens)
            masked_sentence[6] = ' '.join([self.mask_token]*num_object_tokens)
        return masked_sentence
    
    def get_statement_for_verb_detection(self, negate=False, prefix_entities=True):
        masked_statement = self.get_masked_statement(self.pos_tagger.tokenizer.mask_token, negate, prefix_entities)[1:-1]
        masked_statement[3] = ''
        predicate_phrase_start_id = len(' '.join(masked_statement[:3]))
        predicate_phrase_end_id = len(' '.join(masked_statement[:5]))
        phrase_range_ids = {'start': predicate_phrase_start_id, 'end': predicate_phrase_end_id}
        return masked_statement, phrase_range_ids

class HypothesisTriplet:
    def __init__(self, triplet, pos_tagger, tokenizer):
        self.indexed_subject = triplet['subject'][0]
        self.indexed_object = triplet['object'][0]
        self.pos_tagger = pos_tagger
        self.tokenizer = tokenizer
        self.set_clean_triplet_elements(triplet)
        self.verblike_pos_tags = {'VERB', 'AUX'}
    
    def set_clean_triplet_elements(self, triplet):
        self.subject = clean_entity(triplet['subject'][1])
        self.property = clean_entity(triplet['property'])
        self.object = clean_entity(triplet['object'][1])

    def contains_verb(self, predicate):
        return self.verblike_pos_tags in {pos_tag['entity'] for pos_tag in self.pos_tagger(predicate)}
            

    def get_masked_statements_with_mapped_entities(self, verbalisations_variants):
        masked_elements_template = ['']*8
        masked_elements_template[0] = '<s>'

        if not self.contains_verb(clean_entity(f" {self.property}")):
            masked_elements_template[3] = ' <mask>'
        masked_elements_template[4] = f" {self.property}"
        masked_elements_template[7] = '.</s>'
        masked_elements = list()
        for premise_verbalisations_variants in verbalisations_variants:
            for verbalisation_variants in premise_verbalisations_variants:
                entity_mappings = verbalisation_variants['entity_mappings']
                if (self.indexed_subject in entity_mappings) and (self.indexed_object in entity_mappings):
                    masked_elements_variant = masked_elements_template.copy()
                    masked_elements_variant[1:3] = entity_mappings[self.indexed_subject]
                    masked_elements_variant[5:7] = entity_mappings[self.indexed_object]
                    masked_elements.append(masked_elements_variant)
        return masked_elements
