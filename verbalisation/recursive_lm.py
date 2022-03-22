import random

import numpy as np
import torch

from SyntheticNLI.verbalisation.verbalisation import (
    VerbalisationTriplet, 
    replace_triplet_entities, 
    find_entity_phrases,
    clean_entity
    )

class DummyConstraint:
    @staticmethod
    def filter(verbalisation_states, complete_stages, beam_size):
        return verbalisation_states[:beam_size]

class POSConstraint:
    def __init__(self, tokenizer, pos_tagger, constraints_args):
        self.tokenizer = tokenizer
        self.pos_tagger = pos_tagger
        self.constraints_args = constraints_args
        self.verblike_pos_tags = ('VERB', 'AUX')

    def contains_verblike_tag(self, pos_tags):
        for verblike_tag in self.verblike_pos_tags:
            if verblike_tag in pos_tags:
                return True
        return False

    def filter(self, verbalisation_states, complete_stages, beam_size):
        valid_states = list()
        for verbalisation_state in verbalisation_states:
            sentence = self.tokenizer.decode(verbalisation_state['sentence_tokens'][0])
            sentence = clean_entity(sentence.replace('<mask>', '').replace('<s>', ''))
            entity_phrases = find_entity_phrases(
                sentence,
                self.constraints_args['triplet'].subject,
                self.constraints_args['triplet'].property,
                self.constraints_args['triplet'].object,
            )
            if entity_phrases is None:
                continue
            subject_phrase, property_phrase, object_phrase = entity_phrases
            pos_tags = [pos_tag['entity'] for pos_tag in self.pos_tagger(sentence)]
            subject_phrase_words_len = len([w for w in subject_phrase.split(' ') if w != ''])
            property_phrase_words_len = len([w for w in property_phrase.split(' ') if w != ''])
            object_phrase_words_len = len([w for w in object_phrase.split(' ') if w != ''])
            subject_phrase_pos_tags = pos_tags[:subject_phrase_words_len]
            property_phrase_pos_tags = pos_tags[subject_phrase_words_len:subject_phrase_words_len+property_phrase_words_len]
            object_phrase_pos_tags = pos_tags[subject_phrase_words_len+property_phrase_words_len:]
            valid_verbalization = True
            if (0 in complete_stages) and self.contains_verblike_tag(subject_phrase_pos_tags):
                valid_verbalization = False
            if (1 in complete_stages) and (not self.contains_verblike_tag(property_phrase_pos_tags)):
                valid_verbalization = False
            if (2 in complete_stages) and self.contains_verblike_tag(object_phrase_pos_tags):
                valid_verbalization = False
            if valid_verbalization:
                valid_states.append(verbalisation_state)
            if len(valid_states) >= beam_size:
                break
        return valid_states


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
            tokens.extend(part_tokens)
            num_part_tokens = part_tokens.shape[1]
            token_ranges.append((token_start_id, token_start_id + num_part_tokens))
            token_start_id += num_part_tokens
        tokens = torch.cat(tokens).reshape(1, -1)
        positions = torch.where(tokens[0] == self.tokenizer.mask_token_id)[0].numpy()
        assert len(positions) == len(order)
        reordered_positions = positions[order].tolist()
        stages = ([], order)

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
            stages,
            constraint,
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



    def recusive_fill(self, verbalisation_states, positions_list, stages, constraint):
        if len(positions_list) == 0:
            return verbalisation_states
        else:
            modified_verbalisation_states = list()
            for verbalisation_state in verbalisation_states:
                model_logits = self.model(verbalisation_state['sentence_tokens'])['logits'][0][positions_list[0]]
                probs = torch.nn.functional.softmax(model_logits, dim=0)
                #top_k_tokens = torch.topk(model_logits, self.top_k, dim=0).indices.tolist()
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
            complete_stages = stages[0] + [stages[1][0]]
            upcoming_stages = stages[1][1:]
            sorted_states = constraint.filter(sorted_states, complete_stages, self.beam_size)
            print(complete_stages)
            for s in sorted_states:
                print(self.tokenizer.decode(s['sentence_tokens'][0]))
            return self.recusive_fill(
                sorted_states[:self.beam_size], 
                positions_list[1:], 
                (complete_stages, upcoming_stages), 
                constraint
            )

