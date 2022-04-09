import random

import numpy as np
import torch

class DummyConstraint:
    @staticmethod
    def filter(verbalisation_states, beam_size, token_ranges):
        return verbalisation_states[:beam_size]

class POSConstraint:
    def __init__(self, tokenizer, pos_tagger):
        self.tokenizer = tokenizer
        self.pos_tagger = pos_tagger
        self.verblike_pos_tags = {'VERB', 'AUX'}

    #@staticmethod
    def is_fully_filled(self, pos_tags):
        for pos_tag in pos_tags:
            if pos_tag['word'] == '[MASK]':
                return False
        return True

    def contains_verblike_tag(self, pos_tags):
        for pos_tag in pos_tags:
            if pos_tag['entity'] in self.verblike_pos_tags:
                return True
        return False

    def filter(self, verbalisation_states, beam_size, token_ranges):
        valid_states = list()
        for verbalisation_state in verbalisation_states:

            verbalised_sentence_parts = list()
            for token_range in token_ranges[1:-1]:
                verbalisation_part_decoded = self.tokenizer.decode(verbalisation_state['sentence_tokens'][0][token_range[0]:token_range[1]])
                verbalised_sentence_parts.append(verbalisation_part_decoded.replace('<mask>', '[MASK]'))
            verbalised_sentence = ''.join(verbalised_sentence_parts)

            subject_phrase_characters_start = 0
            subject_phrase_characters_end = len(''.join(verbalised_sentence_parts[0:2]))
            predicate_phrase_characters_start = len(''.join(verbalised_sentence_parts[0:2]))
            predicate_phrase_characters_end = len(''.join(verbalised_sentence_parts[0:4]))
            object_phrase_characters_start = len(''.join(verbalised_sentence_parts[0:4]))
            object_phrase_characters_end = len(''.join(verbalised_sentence_parts[0:6]))
            
            subject_phrase_pos_tags = list()
            predicate_phrase_pos_tags = list()
            object_phrase_pos_tags = list()
            for pos_tag in self.pos_tagger(verbalised_sentence):
                if (pos_tag['start'] >= subject_phrase_characters_start) and (pos_tag['end'] <= subject_phrase_characters_end):
                    subject_phrase_pos_tags.append(pos_tag)
                if (pos_tag['start'] >= predicate_phrase_characters_start) and (pos_tag['end'] <= predicate_phrase_characters_end):
                    predicate_phrase_pos_tags.append(pos_tag)
                if (pos_tag['start'] >= object_phrase_characters_start) and (pos_tag['end'] <= object_phrase_characters_end):
                    object_phrase_pos_tags.append(pos_tag)

            valid_verbalization = True
            if self.is_fully_filled(subject_phrase_pos_tags) and self.contains_verblike_tag(subject_phrase_pos_tags):
                valid_verbalization = False
            if self.is_fully_filled(predicate_phrase_pos_tags) and not self.contains_verblike_tag(predicate_phrase_pos_tags):
                valid_verbalization = False
            if self.is_fully_filled(object_phrase_pos_tags) and self.contains_verblike_tag(object_phrase_pos_tags):
                valid_verbalization = False
            if valid_verbalization:
                valid_states.append(verbalisation_state)

            if len(valid_states) >= beam_size:
                break

        return valid_states

class DummyModifier:
    def __init__(self, *args, **kwargs):
        pass
    
    def modify_scores(self, score, *args, **kwargs):
        return score

class Modifier:
    def __init__(self, model, tokenizer, domains, weight=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.weight = weight
        self.modification_vectors = self._get_domain_scores(domains)
        self.iterator_idx = 0
        
    @staticmethod
    def _get_masked_domain_statement(domain_name):
        normalized_domain_name = domain_name.lower()
        return f'<s><mask> is a {normalized_domain_name}.</s>'

    def _get_domain_scores(self, domains):
        domain_scores = list()
        for domian in domains:
            masked_domain_statement = self._get_masked_domain_statement(domian)
            masked_domain_statement_tokens = self.tokenizer.encode(masked_domain_statement, return_tensors="pt", add_special_tokens=False)
            model_logits = self.model(masked_domain_statement_tokens)['logits'][0][1]
            probs = torch.nn.functional.softmax(model_logits, dim=0)
            domain_scores.append(probs)
        return domain_scores      
    
    def modify_scores(self, score):
        modification_vector = self.modification_vectors[self.iterator_idx]
        modified_score = torch.pow(score, 1-self.weight)*torch.pow(modification_vector, self.weight)
        self.iterator_idx += 1
        return modified_score

class RecursiveLM:
    def __init__(self, model, tokenizer, beam_size=20, order='random'):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size

    @staticmethod
    def get_constraint(constraints):
        if constraints is None:
            return DummyConstraint()
        return constraints
        
    def fill_multiple_masks(self, sequence, order, constraint=None):
        constraint = self.get_constraint(constraint)
        tokens = list()
        token_ranges = list()
        token_start_id = 0
        for sequence_part in sequence:
            part_tokens = self.tokenizer.encode(sequence_part,
                                    return_tensors="pt",
                                    add_special_tokens=False)
            num_part_tokens = part_tokens.shape[1]
            if num_part_tokens > 0:
                tokens.extend(part_tokens)
            token_ranges.append((token_start_id, token_start_id + num_part_tokens))
            token_start_id += num_part_tokens
        tokens = torch.cat(tokens).reshape(1, -1)
        positions = torch.where(tokens[0] == self.tokenizer.mask_token_id)[0].numpy()
        assert len(positions) == len(order)
        reordered_positions = positions[order].tolist()

        initial_verbalisation_states = [
            {
                'sentence_tokens': tokens, 
                'filled_tokens': [], 
                'probabilities': [],
            }
        ]
        verbalisations = self.recusive_fill(
            initial_verbalisation_states, 
            reordered_positions,
            constraint,
            token_ranges,
            )
        verbalised_sentences = list()
        filled_words = list()
        for verbalisation in verbalisations:
            verbalised_sentence = list()
            for token_range in token_ranges:
                verbalised_sentence.append(self.tokenizer.decode(verbalisation['sentence_tokens'][0][token_range[0]:token_range[1]]))
            verbalised_sentences.append(verbalised_sentence)
            filled_words.append([self.tokenizer.decode(filled_token) for filled_token in verbalisation['filled_tokens']])
        return verbalised_sentences, filled_words



    def recusive_fill(self, verbalisation_states, positions_list, constraint, token_ranges):
        if len(positions_list) == 0:
            return verbalisation_states
        else:
            modified_verbalisation_states = list()
            for verbalisation_state in verbalisation_states:
                model_logits = self.model(verbalisation_state['sentence_tokens'])['logits'][0][positions_list[0]]
                probs = torch.nn.functional.softmax(model_logits, dim=0)
                sorted_tokens = torch.argsort(probs, descending=True)
                for token in sorted_tokens:
                    modified_tokens = verbalisation_state['sentence_tokens'].detach().clone()
                    modified_tokens[0][positions_list[0]] = token
                    fill_probability = probs[token].detach().numpy().item()
                    modified_verbalisation_states.append(
                        {
                            'sentence_tokens': modified_tokens, 
                            'filled_tokens': verbalisation_state['filled_tokens'] + [token], 
                            'probabilities': verbalisation_state['probabilities'] + [fill_probability]
                        }
                    )
            sorted_states = sorted(modified_verbalisation_states, key=lambda x: np.mean(x['probabilities']), reverse=True)
            sorted_states = constraint.filter(sorted_states, self.beam_size, token_ranges)
            return self.recusive_fill(
                sorted_states[:self.beam_size], 
                positions_list[1:],
                constraint,
                token_ranges,
            )

