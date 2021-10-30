from typing import Any, Dict

import yaml


class ConfigLoader:
    def __init__(
        self, local_root_path: str, config_root_path: str = "configs"
    ) -> None:
        self.local_root_path = local_root_path
        self.config_root_path = config_root_path

    def load(
        self, pipeline_type: str, exp_id: str, default_exp_id: str
    ) -> Dict[str, Any]:
        config = self.__load_config_from_yaml(
            pipeline_type=pipeline_type, exp_id=exp_id
        )
        default_config = self.__load_config_from_yaml(
            pipeline_type=pipeline_type, exp_id=default_exp_id
        )
        self.__fill_config_by_default_config(
            config_dict=config, default_config_dict=default_config
        )
        return config

    def __config_filepath_with_local_root(self, pipeline_type: str, exp_id: str) -> str:
        return self.__filepath_with_local_root(
            self.__config_filepath_from_root(pipeline_type=pipeline_type, exp_id=exp_id)
        )

    def __config_filepath_from_root(self, pipeline_type: str, exp_id: str) -> str:
        return f"{self.config_root_path}/{pipeline_type}/{exp_id}.yml"

    def __filepath_with_local_root(self, filepath_from_root: str) -> str:
        return f"{self.local_root_path}/{filepath_from_root}"

    def __load_config_from_yaml(
        self, pipeline_type: str, exp_id: str
    ) -> Dict[str, Any]:
        yaml_filename = self.__config_filepath_with_local_root(
            pipeline_type=pipeline_type, exp_id=exp_id
        )
        with open(yaml_filename, "r") as fin:
            config: Dict[str, Any] = yaml.load(fin, Loader=yaml.FullLoader)

        return config

    def __fill_config_by_default_config(
        self,
        config_dict: Dict[str, Any],
        default_config_dict: Dict[str, Any],
    ) -> None:
        for (d_key, d_value) in default_config_dict.items():
            if d_key not in config_dict:
                config_dict[d_key] = d_value
            elif isinstance(d_value, dict):
                self.__fill_config_by_default_config(config_dict[d_key], d_value)

        default_config_keys = set(default_config_dict.keys())
        config_keys = set(config_dict.keys())
        only_config_keys = config_keys - default_config_keys
        if len(only_config_keys):
            raise Exception(f"keys {only_config_keys} do not exist in default config.")
