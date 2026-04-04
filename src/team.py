#!/usr/bin/python
from typing import Dict, List

class Team:
    # Gemeinsame (globale) Mappings für alle Instanzen
    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}

    # Flag, ob die Initialisierung schon passiert ist
    _initialized = False

    def __init__(self):
        # Initialisierung nur beim ersten Erstellen einer Instanz
        if not Team._initialized:
            raise TypeError("This class cannot be instantiated, because it is not initialized. Please call 'initialize' first to set up the team mappings.")

    @classmethod
    def initialize(cls, default_teams: List[str]):
        """Definiert alle Teams einmalig beim ersten Instanziieren."""
        Team._initialized = True
        Team.name_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(default_teams)}
        Team.id_to_name: Dict[int, str] = {idx: name for idx, name in enumerate(default_teams)}

    def add_team(self, name: str, team_id: int):
        if name in Team.name_to_id:
            raise ValueError(f"Teamname '{name}' existiert bereits.")
        if team_id in Team.id_to_name:
            raise ValueError(f"Team-ID '{team_id}' existiert bereits.")

        Team.name_to_id[name] = team_id
        Team.id_to_name[team_id] = name

    @classmethod
    def Id(cls, name: str):
        return cls.name_to_id.get(name)

    @classmethod
    def Name(cls, team_id: int):
        return cls.id_to_name.get(team_id)

    @classmethod
    def AllIds(cls):
        return list(cls.id_to_name.keys())

    @classmethod
    def AllNames(cls):
        return list(cls.name_to_id.keys())


if __name__ == "__main__":
    # Beispiel für die Initialisierung der Teams
    default_teams = ['EV Zug', 'ZSC Lions', 'HC Davos', 'HC Ajoie']
    Team.initialize(default_teams)
    # Beispiel für die Verwendung der Team-Klasse
    print(Team.Id('EV Zug'))  # Ausgabe: 0
    print(Team.Name(1))       # Ausgabe: 'ZSC Lions'

    print("All team IDs:", Team.AllIds())       # Ausgabe: [0, 1, 2, 3]
    print("All team names:", Team.AllNames())   # Ausgabe: ['EV Zug', 'ZSC Lions', 'HC Davos', 'HC Ajoie']