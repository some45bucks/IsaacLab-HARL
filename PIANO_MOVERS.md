# Registering a New Environment in Isaac Lab

Isaac Lab allows you to register custom environments in the Gym framework, enabling efficient use with Gym's `make` command. This guide provides steps to register a new environment, using the example of the `DirectEnvRL`-based Piano Movers environment.

---

## Environment Registration Example

The environment `DirectEnvRL` is used as the base for Piano Movers, and it is registered in the Gym environments through the `__init__.py` file. To better understand the process, you can explore the following example folder:

```
IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/anymal_c_piano_movers
```

This folder contains the environment code and the necessary registration logic. To register your custom environment, follow these steps:

---

## Steps to Register a New Environment

### 1. **Modify Extensions**

If you are adding a new `DirectEnvRL` environment, place your code in the following directory:

```
IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/
```

### 2. **Update `__init__.py`**

Ensure that your new environment is registered in the Gym framework by adding the appropriate registration logic to the `__init__.py` file in the corresponding directory.

### 3. **Reinstall the Extensions Package**

After making changes to the extensions, you need to reinstall the `omni.isaac.lab_tasks` package. Use the following command:

```bash
python -m pip install -e source/extensions/omni.isaac.lab_tasks/
```

This command updates the extension for Isaac Sim, ensuring your changes are properly registered.

---

## Running the Registered Environment

To run a script that uses the newly registered environment, use the following.

```bash
python source/isaacdev/envs/piano_movers_direct_env.py --task Isaac-Piano-Movers-Flat-Anymal-C-Direct-v0 --num_envs 2
```

Replace `piano_movers_direct_env.py` with the path to your environment script.

---

## Summary

- **Location of Environment Code**:
  - `IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/`
- **Reinstall Extensions**:
  - `python -m pip install -e source/extensions/omni.isaac.lab_tasks/`
- **Run the Environment**:
  - `./isaaclab.sh -p <path_to_your_script>`

By following these steps, you can successfully register and run your custom environments in Isaac Lab.

# TODO

- The environment only passes in one action for each robot at a time when you instantiate them in a single environment, which
is an issue because the global actions that we store keep track of the actions for all the robots, so need to figure out a 
way to update the global actions or change the environment code so that is processes all the actions for all the robots at the same time.