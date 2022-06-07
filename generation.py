from abc import ABC, abstractmethod
from collections import defaultdict
import random
import re

import numpy as np
import owlready2 as owl


class Triplet:
    def __init__(self, subject, _property, _object=None):
        self.subject = subject
        self._property = _property
        if _object is not None:
            self._object = _object

    def property_is_functional(self):
        _object = getattr(self.subject, self._property.name)
        return not isinstance(_object, list)

    def add_object(self, _object):
        if self.property_is_functional():
            setattr(self.subject, self._property.name, _object)
        else:
            current_objects = getattr(self.subject, self._property.name)
            current_objects.append(_object)
        self._object = getattr(self.subject, self._property.name)

    def get_object_entities(self):
        if self.property_is_functional():
            return [self._object]
        else:
            return self._object

    def delete_entity(self, entity, initial_individuals):
        if entity not in initial_individuals:
            owl.destroy_entity(entity)

    def reset_property(self):
        if self.property_is_functional():
            setattr(self.subject, self._property.name, None)
        else:
            setattr(self.subject, self._property.name, [])

    def delete(self, initial_individuals):
        self.delete_entity(self.subject, initial_individuals)
        self.reset_property()
        for object_entity in self.get_object_entities():
            self.delete_entity(object_entity, initial_individuals)

    @staticmethod
    def get_entity_class_name(entity):
        return str(type(entity)).split('.')[-1]

    def get_entity_names(self, entity):
        return entity.name, self.get_entity_class_name(entity)

    def parsed(self):
        if self.property_is_functional():
            return {(
                self.get_entity_names(self.subject), 
                self.get_entity_names(self._property), 
                self.get_entity_names(self._object),
                )}
        else:
            return {
                (
                    self.get_entity_names(self.subject), 
                    self.get_entity_names(self._property), 
                    self.get_entity_names(_object),
                )
                for _object in self._object
            }

    def _get_name(self, entity):
        if isinstance(entity, list):
            return entity[0]
        try:
            return entity.name
        except AttributeError:
            return entity

    def __str__(self):
        return " ".join(
            [self.subject.name, self._property.name]
            + [o.name for o in self.get_object_entities()]
        )


class WorldGenerator(ABC):
    def __init__(self, ontology_path: str):
        self.ontology_path = ontology_path
        self.world = owl.World()
        self.ontology = self._get_ontology()
        self.classes = self._get_classes()
        self.properties = self._get_properties()
        self.initial_individuals = self._get_individuals()
        self.owl_triplets = self._get_triplets()
        self.invalid_classes = self._get_invalid_classes()
        self.property_domain_range_map, self.domain_property_map, self.range_property_map = self._get_property_maps()
        self.valid_properties = self._get_valid_properties()
        self.valid_classes = self._get_valid_classes()

    def _get_ontology(self):
        return self.world.get_ontology(self.ontology_path).load()

    def _reset_ontology(self):
        self.ontology = self._get_ontology()

    def _get_invalid_classes(self):
        invalid_classes = set()
        for _class in self.classes:
            try:
                test_individual = _class()
                owl.destroy_entity(test_individual)
            except AttributeError:
                invalid_classes.add(_class)
        self._reset_ontology()
        return invalid_classes

    def _get_property_domains_and_ranges(self):
        filled_properties = list()
        for _property in self.ontology.object_properties():
            domains, ranges = self.get_recurrent_domain_and_range(_property, [], [])
            filled_properties.append((_property, domains, ranges))
        return filled_properties

    def get_recurrent_domain_and_range(self, entity, domain, _range):
        if len(domain) == 0:
            domain = entity.domain
        if len(_range) == 0:
            _range = entity.range
        if len(domain) > 0 and len(_range) > 0:
            return domain, _range
        else:
            parents = self.ontology.get_parents_of(entity)
            if len(parents) > 0:
                return self.get_recurrent_domain_and_range(parents[0], domain, _range)
            else:
                return domain, _range

    def _get_subclasses(self, classes):
        recursive_subclasses = set(classes) - {None}
        for _class in classes:
            if _class is None:
                continue
            for subclass in _class.subclasses():
                if subclass != _class:
                    recursive_subclasses = recursive_subclasses.union(self._get_subclasses([subclass]))
        return list(recursive_subclasses)

    def _get_property_maps(self):
        property_domain_range_map = dict()
        domain_property_map = dict()
        range_property_map = dict()
        for _property in self.ontology.object_properties():
            domains, ranges = self.get_recurrent_domain_and_range(_property, [], [])
            full_domain = self._get_subclasses(domains)
            full_range = self._get_subclasses(ranges)
            valid_domain = [_class for _class in full_domain if _class not in self.invalid_classes]
            valid_range = [_class for _class in full_range if _class not in self.invalid_classes]
            if len(valid_domain) > 0 and len(valid_range) > 0:
                property_domain_range_map[_property] = (valid_domain, valid_range)
                for domain_class in valid_domain:
                    if domain_class not in domain_property_map:
                        domain_property_map[domain_class] = list()
                    domain_property_map[domain_class].append(_property)
                for range_class in valid_range:
                    if range_class not in range_property_map:
                        range_property_map[range_class] = list()
                    range_property_map[range_class].append(_property)
        return property_domain_range_map, domain_property_map, range_property_map

    def _get_valid_properties(self):
        return list(self.property_domain_range_map.keys())

    def _get_valid_classes(self):
        return list(set(self.domain_property_map.keys()).union(self.range_property_map.keys()))

    @abstractmethod
    def generate_world(self):
        pass

    def _get_classes(self):
        return list(self.ontology.classes())

    def _get_individuals(self):
        return set(self.ontology.individuals())

    def _get_properties(self):
        return list(self.ontology.object_properties())

    def _get_triplets(self):
        triplets = list()
        for _object in self.ontology.individuals():
            for _property in _object.get_properties():
                if _property.name not in {"schema", "label", "comment"}:
                    subject = getattr(_object, _property.name)
                    triplets.append(Triplet(_object, _property, subject))
        return triplets


    def parse_world(self, triplets):
        parsed_triplets = set()
        for triplet in triplets:
            if triplet._property in self.property_domain_range_map:
                for parsed_triplet in triplet.parsed():
                    parsed_triplets.add(parsed_triplet)
        return parsed_triplets

    def run_inference(self):
        with self.ontology:
            # owl.sync_reasoner_pellet(infer_property_values=True)
            owl.sync_reasoner(infer_property_values=True, debug = False)
            # try:
            #     owl.sync_reasoner_pellet(infer_property_values=True)
            # except:
            #     return None
        inferred_triplets = self._get_triplets()
        return inferred_triplets

class IncrementalGenerator(WorldGenerator):
    def __init__(self, ontology_path: str, num_triplets: int, new_graph_prob: float, extend_graph_prob: float, add_edge_prob: float):
        super(IncrementalGenerator, self).__init__(ontology_path)
        self.actions = ['new_graph', 'extend_graph', 'add_edge']
        self.num_triplets = num_triplets
        self.action_probability_distribution = [new_graph_prob, extend_graph_prob, add_edge_prob]

    def generate_world(self):
        self._reset_ontology()
        initial_triplets = self._generate_premises()
        post_inference_triplets = self.run_inference()
        parsed_initial_triplets = self.parse_world(initial_triplets)
        inferred_triplets = (
            self.parse_world(post_inference_triplets) - parsed_initial_triplets
        )
        for triplet in post_inference_triplets:
            triplet.delete(self.initial_individuals)
        return parsed_initial_triplets, inferred_triplets

    def _generate_premises(self):
        individuals, triplets = self.create_new_graph(set(), dict())
        num_actions = self.num_triplets - 1
        actions = np.random.choice(self.actions, num_actions, p=self.action_probability_distribution)
        for action in actions:
            if action == 'new_graph':
                individuals, triplets = self.create_new_graph(individuals, triplets)
            elif action == 'extend_graph':
                individuals, triplets = self.extend_graph(individuals, triplets)
            elif action == 'add_edge':
                individuals, triplets = self.add_edge(individuals, triplets)
            else:
                raise
        return {triplet for key, triplet in triplets.items()}


    def create_new_graph(self, individuals, triplets):
        predicate = random.choice(list(self.property_domain_range_map.keys()))
        domain, _range = self.property_domain_range_map[predicate]
        subject_class = random.choice(domain)
        object_class = random.choice(_range)
        subject = subject_class()
        _object = object_class()
        subject = subject_class()
        triplet = Triplet(subject, predicate, _object)
        triplet.add_object(_object)
        individuals = individuals.union({subject, _object})
        triplets[(subject, predicate)] = triplet
        return individuals, triplets

    def extend_graph(self, individuals, triplets):
        node_to_extend = random.sample(individuals, 1)[0]
        existing_node_role = random.choice(["subject", "object"])
        existing_node_class = type(node_to_extend)
        if existing_node_role == "subject" and existing_node_class in self.domain_property_map:
            predicates = list(self.domain_property_map[existing_node_class])
            shuffled_predicates = random.sample(predicates, len(predicates))
            for predicate in shuffled_predicates:
                if (node_to_extend, predicate) in triplets:
                    triplet = triplets[(node_to_extend, predicate)]
                    if triplet.property_is_functional():
                        continue
                    else:
                        _, _range = self.property_domain_range_map[predicate]
                        object_class = random.choice(_range)
                        _object = object_class()
                        triplet.add_object(_object)
                        individuals = individuals.union({_object})
                        return individuals, triplets
                else:
                    _, _range = self.property_domain_range_map[predicate]
                    object_class = random.choice(_range)
                    _object = object_class()
                    triplet = Triplet(node_to_extend, predicate, _object)
                    triplet.add_object(_object)
                    individuals = individuals.union({_object})
                    triplets[(node_to_extend, predicate)] = triplet
                    return individuals, triplets
            return individuals, triplets
        elif existing_node_class in self.range_property_map:
            predicate = random.choice(list(self.range_property_map[existing_node_class]))
            domain, _ = self.property_domain_range_map[predicate]
            subject_class = random.choice(domain)
            subject = subject_class()
            triplet = Triplet(subject, predicate, node_to_extend)
            triplet.add_object(node_to_extend)
            individuals = individuals.union({subject})
            triplets[(subject, predicate)] = triplet
            return individuals, triplets
        else:
            return individuals, triplets

    def add_edge(self, individuals, triplets):
        shuffled_individuals = random.sample(individuals, len(individuals))
        for candidate_subject in shuffled_individuals:
            subject_class = type(candidate_subject)
            if subject_class not in self.domain_property_map:
                continue
            predicates = self.domain_property_map[subject_class]
            shuffled_predicates = random.sample(predicates, len(predicates))
            candidate_objects =  random.sample(individuals - {candidate_subject}, len(individuals - {candidate_subject}))
            for predicate in shuffled_predicates:
                _, _range = self.property_domain_range_map[predicate]
                for candidate_object in candidate_objects:
                    if type(candidate_object) in _range:
                        if (candidate_subject, predicate) in triplets:
                            triplet = triplets[(candidate_subject, predicate)]
                            if triplet.property_is_functional():
                                continue
                            else:
                                if candidate_object in triplet._object:
                                    continue
                                else:
                                    triplet.add_object(candidate_object)
                                    return individuals, triplets
                        else:
                            triplet = Triplet(candidate_subject, predicate, candidate_object)
                            triplet.add_object(candidate_object)
                            triplets[(candidate_subject, predicate)] = triplet
                            return individuals, triplets
        return individuals, triplets


def convert_to_json(worlds):
    return [
        {
            "premise": convert_world(initial_world),
            "hypothesis": convert_world(inferred_world),
        }
        for initial_world, inferred_world in worlds
    ]


def convert_world(world):
    return [convert_triplet(triplet) for triplet in world]


def convert_triplet(triplet):
    subject_names, property_names, object_names = triplet
    subject_name, subject_class_name = subject_names
    property_name, _ = property_names
    object_name, object_class_name = object_names
    return {
        "subject": (subject_name, split_camelcase(subject_class_name)),
        "property": split_camelcase(property_name),
        "object": (object_name, split_camelcase(object_class_name)),
    }


def split_camelcase(_string):
    splitted_camelcase = re.sub(
        "([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", _string)
    ).split()
    rejoined = " ".join(splitted_camelcase)
    return rejoined
