from functions.greedyfool import greedyfool_attack_white
from functions.B3D import B3D_attack_black
from functions.homotopy import homotopy_attack_white
from functions.perturbation_factorization import perturbation_attack_black
from functions.sparse_imperceptible import cornersearch_attack_black, PGD_attack_white


class Attack(object):
    def __init__(self, attack_algorithm) -> None:
        self.mapping = {'B3D_w': B3D_attack_black,
            'greedyfool_w': greedyfool_attack_white, 
            'cornersearch_w': cornersearch_attack_black, 
            'PGD_attack_w': PGD_attack_white, 
            'homotopy_w': homotopy_attack_white, 
            'perturbation_w': perturbation_attack_black}
        self.set_attack_method(attack_algorithm)
    
    def set_attack_method(self, attack_algorithm):
        self.attack = self.mapping[attack_algorithm]


def attack_factory(attack_algorithm):
    return Attack(attack_algorithm)
