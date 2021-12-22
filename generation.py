from abc import ABC, abstractmethod
from collections import defaultdict
import random
import re

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

    def parsed(self):
        if self.property_is_functional():
            return {(self.subject.name, self._property.name, self._object.name)}
        else:
            return {
                (self.subject.name, self._property.name, _object.name)
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


def parse_world(triplets):
    parsed_triplets = set()
    for triplet in triplets:
        for parsed_triplet in triplet.parsed():
            parsed_triplets.add(parsed_triplet)
    return parsed_triplets


def get_subclasses(entity):
    recursive_subclasses = [entity]
    for subclass in entity.subclasses():
        if subclass != entity:
            recursive_subclasses.extend(get_subclasses(subclass))
    return recursive_subclasses


class WorldGenerator(ABC):
    def __init__(self):
        self.ontology = self._get_ontology()
        self.classes = self._get_classes()
        self.initial_individuals = self._get_individuals()
        self.properties = self._get_properties()
        self.owl_triplets = self._get_triplets()
        self.domain_property_map, self.property_range_map = self._get_property_maps()
        self.property_domain_range_map = self.get_property_domain_range_map()

    @staticmethod
    def _get_ontology():
        return owl.get_ontology("http://kb.openrobots.org/").load()
        # return owl.World().get_ontology("http://kb.openrobots.org/").load()

    def _get_property_domains_and_ranges(self):
        filled_properties = list()
        for _property in self.ontology.object_properties():
            domains, ranges = self.get_recurrent_domain_and_range(_property, [], [])
            filled_properties.append((_property, domains, ranges))
        return filled_properties

    def _get_property_maps(self):
        filled_properties = self._get_property_domains_and_ranges()
        domain_property_map = defaultdict(list)
        property_range_map = defaultdict(list)
        for _property, domains, ranges in filled_properties:
            for domain in domains:
                domain_property_map[domain].append(_property)
            for _range in ranges:
                property_range_map[_property].append(_range)
        return domain_property_map, property_range_map

    def get_property_domain_range_map(self):
        filled_properties = self._get_property_domains_and_ranges()
        return {
            _property: (domains, ranges)
            for _property, domains, ranges in filled_properties
        }

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

    def run_inference(self):
        with self.ontology:
            try:
                owl.sync_reasoner_pellet(infer_property_values=True)
            except:
                return None
        inferred_triplets = self._get_triplets()
        return inferred_triplets


class SubjectBasedGenerator(WorldGenerator):
    def generate_world(self, num_subjects, num_properties):
        subjects = [self.generate_subject() for _ in range(num_subjects)]
        for _ in range(num_properties):
            self.ascribe_property(subjects)

    def generate_subject(self):
        _class = random.choice(self.classes)
        return _class()

    def generate_property(self, subject):
        valiable_properties = self.domain_property_map[subject]
        if len(valiable_properties) == 0:
            return None
        return random.choice(valiable_properties)

    def generate_object(self, _property):
        valiable_ranges = self.property_range_map[subject]
        if len(valiable_ranges) == 0:
            return None
        _range = random.choice(valiable_ranges)
        valiable_objects = list(self.ontology.get_instances_of(_range))
        if len(valiable_objects) == 0:
            return None
        return random.choice(valiable_objects)

    def ascribe_property(self, individuals):
        _object = None
        while _object == None:
            subject = random.choice(individuals)
            _property = self.generate_property(subject)
            if _property is None:
                continue
            _object = self.generate_object(_property)
        try:
            setattr(subject, _property.name, _object)
        except:
            setattr(subject, _property.name, [_object])


class PropertyBasedGenerator(WorldGenerator):
    def _generate_property(self):
        while True:
            _property = random.choice(self.properties)
            domains, ranges = self.property_domain_range_map[_property]
            if len(domains) == 0 or len(ranges) == 0:
                continue

            # valiable_subjects = list(self.ontology.get_instances_of(domains[0]))
            # if len(valiable_subjects) > 0:
            #     subject = random.choice(valiable_subjects)
            # else:
            valiable_subject_classes = get_subclasses(domains[0])
            subject = None
            for _ in range(10):
                subject_class = random.choice(valiable_subject_classes)
                try:
                    subject = subject_class()
                except:
                    pass
                if subject is not None:
                    break

            # valiable_objects = [i for i in self.ontology.get_instances_of(ranges[0]) if i != subject]
            # if len(valiable_objects) > 0:
            #     _object = random.choice(valiable_objects)
            # else:
            valiable_object_classes = get_subclasses(ranges[0])
            _object = None
            for _ in range(10):
                object_class = random.choice(valiable_object_classes)
                try:
                    _object = object_class()
                except:
                    pass
                if _object is not None:
                    break
            if subject is None or _object is None:
                continue

            triplet = Triplet(subject, _property, _object)
            triplet.add_object(_object)
            return triplet

    def modify_triplets(self, inferred_triplets):
        triplets = list()
        for triplet in inferred_triplets:
            _property = triplet._property
            domains, ranges = self.property_domain_range_map[_property]
            if (len(domains) == 0 or len(ranges) == 0) or random.random() > 0.5:
                triplets.append(triplet)
                continue

            valiable_subjects = [
                i
                for i in self.ontology.get_instances_of(domains[0])
                if i != triplet.subject
            ]
            if len(valiable_subjects) > 0:
                subject = random.choice(valiable_subjects)
            else:
                subject = domains[0]()

            valiable_objects = [
                i for i in self.ontology.get_instances_of(ranges[0]) if i != subject
            ]
            if len(valiable_objects) > 0:
                _object = random.choice(valiable_objects)
            else:
                _object = ranges[0]()

            modyfied_triplet = Triplet(subject, _property, _object)
            modyfied_triplet.add_object(_object)
            triplets.append(modyfied_triplet)
            print(triplet)
            print(modyfied_triplet)
        return triplets

    def generate_world(self, num_properties):
        initial_triplets = [self._generate_property() for _ in range(num_properties)]
        post_inference_triplets = self.run_inference()
        parsed_initial_triplets = parse_world(initial_triplets)
        inferred_triplets = (
            parse_world(post_inference_triplets) - parsed_initial_triplets
        )
        for triplet in post_inference_triplets:
            triplet.delete(self.initial_individuals)
        return parsed_initial_triplets, inferred_triplets


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
    subject, _property, _object = triplet
    return {
        "subject": split_camelcase(subject),
        "property": split_camelcase(_property),
        "object": split_camelcase(_object),
    }


def split_camelcase(_string):
    splitted_camelcase = re.sub(
        "([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", _string)
    ).split()
    rejoined = " ".join(splitted_camelcase)
    return rejoined
