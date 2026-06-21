from setuptools import find_packages, setup
import os

package_name = 'adapt_x'

data_files = [
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]


def package_files(data_files, directory_list):

    paths_dict = {}

    for directory in directory_list:

        for (path, _, filenames) in os.walk(directory):

            for filename in filenames:

                file_path = os.path.join(path, filename)
                install_path = os.path.join('share', package_name, path)

                if install_path in paths_dict.keys():
                    paths_dict[install_path].append(file_path)

                else:
                    paths_dict[install_path] = [file_path]

    for key in paths_dict.keys():
        data_files.append((key, paths_dict[key]))

    return data_files


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=package_files(data_files, ['launch/', 'jars/', 'config/']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ega',
    maintainer_email='45923444+EGAlberts@users.noreply.github.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'adaptation_engine = adapt_x.adaptation_engine:main',
            'tactical_adaptation_node = adapt_x.tactical_adaptation_node:main',
            'fmlearning = adapt_x.adaptation_engine_fm:main',
            'rem_learning = adapt_x.adaptation_engine_policy:main',
            'test_adap_cc = adapt_x.test_action_server:main',
            'test_adap_cc_client = adapt_x.test_action_client:main',
            'trigger_service = adapt_x.test_trigger_service:main',
            'test_monitor = adapt_x.test_monitor:main',
            'setup_node = adapt_x.setup_node:main',
            'adaptation_proxy = adapt_x.proxy_action_server:main',
            'lifecycle_manager = adapt_x.lifecycle_manager:main',
        ],
    },
)
