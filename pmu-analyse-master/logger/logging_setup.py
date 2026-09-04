import os
import tempfile
from logging.config import fileConfig

_CONFIGURED = False


def setup_logging():
    """Configure le logging à partir de logging_config.ini.

    Utilise des chemins absolus (calculés à partir de l'emplacement de ce
    fichier) au lieu de chemins relatifs, pour que ça fonctionne quel que
    soit le répertoire courant depuis lequel le code est lancé (CLI locale,
    `streamlit run`, Streamlit Community Cloud, tests...).

    Le fichier de log est écrit dans le répertoire temporaire du système :
    sur Streamlit Cloud le système de fichiers est éphémère de toute façon,
    et ça évite tout souci de droits d'écriture dans le repo.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'logging_config.ini')
    log_file = os.path.join(tempfile.gettempdir(), 'pmu_analyser.log')

    fileConfig(
        config_path,
        defaults={'logfilename': log_file.replace('\\', '\\\\')},
        disable_existing_loggers=False,
    )
    _CONFIGURED = True
