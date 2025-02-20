"""
    Diagnostic wind classes
    =======================

    Classes defining wind fields diagnostics.

    Description of the classes
    --------------------------

    * :class:`AtmosphericWindDiagnostic`: General base class for atmospheric wind diagnostic.
    * :class:`LowerLayerAtmosphericVWindDiagnostic`: Diagnostic giving the lower layer atmospheric V wind fields :math:`\\partial_x \\psi^3_{\\rm a}`.
    * :class:`LowerLayerAtmosphericUWindDiagnostic`: Diagnostic giving the lower layer atmospheric U wind fields :math:`- \\partial_y \\psi^3_{\\rm a}`.
    * :class:`MiddleAtmosphericVWindDiagnostic`: Diagnostic giving the middle atmospheric V wind fields :math:`\\partial_x \\psi_{\\rm a}`.
    * :class:`MiddleAtmosphericUWindDiagnostic`: Diagnostic giving the middle atmospheric U wind fields :math:`- \\partial_y \\psi_{\\rm a}`.
    * :class:`UpperLayerAtmosphericVWindDiagnostic`: Diagnostic giving the upper layer atmospheric V wind fields :math:`\\partial_x \\psi^1_{\\rm a}`.
    * :class:`UpperLayerAtmosphericUWindDiagnostic`: Diagnostic giving the upper layer atmospheric U wind fields :math:`- \\partial_y \\psi^1_{\\rm a}`.
    * :class:`LowerLayerAtmosphericWindIntensityDiagnostic`: Diagnostic giving the lower layer atmospheric wind intensity fields.
    * :class:`MiddleAtmosphericWindIntensityDiagnostic`: Diagnostic giving the middle atmospheric wind intensity fields.
    * :class:`UpperLayerAtmosphericWindIntensityDiagnostic`: Diagnostic giving the upper layer atmospheric wind intensity fields.

"""

import warnings

import numpy as np
import matplotlib.pyplot as plt

from qgs.diagnostics.base import FieldDiagnostic


class AtmosphericWindDiagnostic(FieldDiagnostic):
    """General base class for atmospheric wind fields diagnostic.
    Provide a spatial gridded representation of the fields.
    This is an `abstract base class`_, it must be subclassed to create new diagnostics!

    .. _abstract base class: https://docs.python.org/3/glossary.html#term-abstract-base-class

    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.
    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        if not hasattr(self, 'type'):
            self.type = None

        FieldDiagnostic.__init__(self, model_params, dimensional)
        self._configure(delta_x=delta_x, delta_y=delta_y)

        self._plot_units = r" (in " + r'm s$^{-1}$' + r")"
        self._default_plot_kwargs['cmap'] = plt.get_cmap('hsv_r')
        self._color_bar_format = False

    def _compute_grid(self, delta_x=None, delta_y=None):

        if delta_x is None:
            ams = self._model_params.ablocks
            if ams is None:
                warnings.warn("AtmosphericWindDiagnostic: Unable to configure the grid automatically. Atmospheric wavenumbers information not " +
                              "present in the model's parameters ! Please call the compute_grid method with the delta_x and delta_y parameters.")
                return 1
            xwn = [ams[i][0] for i in range(len(ams))]
            mxwn = max(xwn)
            n_point_x = 4 * mxwn + 2
        else:
            n_point_x = int(np.ceil((2 * np.pi / self._model_params.scale_params.n) / delta_x) + 1)

        if delta_y is None:
            ams = self._model_params.ablocks
            if ams is None:
                warnings.warn("AtmosphericWindDiagnostic: Unable to configure the grid automatically. Atmospheric wavenumbers information not " +
                              "present in the model's parameters ! Please call the compute_grid method with the delta_x and delta_y parameters.")
                return 1
            ywn = [ams[i][1] for i in range(len(ams))]
            mywn = max(ywn)
            n_point_y = 4 * mywn + 2
        else:
            n_point_y = int(np.ceil(np.pi / delta_y) + 1)

        x = np.linspace(0., 2 * np.pi / self._model_params.scale_params.n, n_point_x)
        y = np.linspace(0., np.pi, n_point_y)
        self._X, self._Y = np.meshgrid(x, y)

    def _configure(self, delta_x=None, delta_y=None):

        self._compute_grid(delta_x, delta_y)

        if self.type == "V":
            dx_basis = self._model_params.atmospheric_basis.x_derivative
            grid_dx_basis = list()

            for func in dx_basis.num_functions():
                grid_dx_basis.append(func(self._X, self._Y))

            # Check for cases where the symbolic derivative is the 0 function
            for i in range(len(grid_dx_basis)):
                if not hasattr(grid_dx_basis[i], 'data'):
                    grid_dx_basis[i] = np.full_like(self._X, grid_dx_basis[i])

            self._grid_basis = np.array(grid_dx_basis)
        elif self.type == "U":
            dy_basis = self._model_params.atmospheric_basis.y_derivative
            grid_dy_basis = list()

            for func in dy_basis.num_functions():
                grid_dy_basis.append(func(self._X, self._Y))

            # Check for cases where the symbolic derivative is the 0 function
            for i in range(len(grid_dy_basis)):
                if not hasattr(grid_dy_basis[i], 'data'):
                    grid_dy_basis[i] = np.full_like(self._X, grid_dy_basis[i])

            self._grid_basis = np.array(grid_dy_basis)
        elif self.type is None:
            warnings.warn("AtmosphericWindDiagnostic: Basis type note specified." +
                          " Unable to configure the diagnostic properly.")
            return 1
        else:
            self._grid_basis = None


class LowerLayerAtmosphericVWindDiagnostic(AtmosphericWindDiagnostic):
    """Diagnostic giving the lower layer atmospheric V wind fields :math:`\\partial_x \\psi^3_{\\rm a}`.
    Computed as :math:`\\partial_x \\psi^3_{\\rm a} = \\partial_x \\psi_{\\rm a} - \\partial_x \\theta_{\\rm a}` where :math:`\\psi_{\\rm a}` and :math:`\\theta_{\\rm a}` are respectively the barotropic and baroclinic streamfunctions.
    See also the :ref:`files/model/atmosphere:Atmospheric component` and :ref:`files/model/oro_model:Mid-layer equations and the thermal wind relation` sections.

    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    delta_x: float, optional
        Spatial step in the zonal direction `x` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    delta_y: float, optional
        Spatial step in the meridional direction `y` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    dimensional: bool, optional
        Indicate if the output diagnostic must be dimensionalized or not.
        Default to `True`.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        self.type = "V"

        AtmosphericWindDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        self._plot_title = r'Atmospheric V wind in the lower layer'

    def _get_diagnostic(self, dimensional):

        natm = self._model_params.nmod[0]
        psi = np.swapaxes(self._data[:natm, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)
        theta = np.swapaxes(self._data[natm:2*natm, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)

        psi3 = psi - theta

        if dimensional:
            self._diagnostic_data = psi3 * self._model_params.streamfunction_scaling / self._model_params.scale_params.L
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = psi3
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class LowerLayerAtmosphericUWindDiagnostic(AtmosphericWindDiagnostic):
    """Diagnostic giving the lower layer atmospheric U wind fields :math:`- \\partial_y \\psi^3_{\\rm a}`.
    Computed as :math:`- \\partial_y \\psi^3_{\\rm a} = - \\partial_y \\psi_{\\rm a} + \\partial_y \\theta_{\\rm a}` where :math:`\\psi_{\\rm a}` and :math:`\\theta_{\\rm a}` are respectively the barotropic and baroclinic streamfunctions.
    See also the :ref:`files/model/atmosphere:Atmospheric component` and :ref:`files/model/oro_model:Mid-layer equations and the thermal wind relation` sections.

    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    delta_x: float, optional
        Spatial step in the zonal direction `x` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    delta_y: float, optional
        Spatial step in the meridional direction `y` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    dimensional: bool, optional
        Indicate if the output diagnostic must be dimensionalized or not.
        Default to `True`.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        self.type = "U"

        AtmosphericWindDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        self._plot_title = r'Atmospheric U wind in the lower layer'

    def _get_diagnostic(self, dimensional):

        natm = self._model_params.nmod[0]
        psi = np.swapaxes(self._data[:natm, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)
        theta = np.swapaxes(self._data[natm:2*natm, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)

        psi3 = psi - theta

        if dimensional:
            self._diagnostic_data = - psi3 * self._model_params.streamfunction_scaling / self._model_params.scale_params.L
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = - psi3
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class MiddleAtmosphericVWindDiagnostic(AtmosphericWindDiagnostic):
    """Diagnostic giving the middle atmospheric V wind fields :math:`\\partial_x \\psi_{\\rm a}`
    where :math:`\\psi_{\\rm a}` is the barotropic streamfunction.
    See also the :ref:`files/model/atmosphere:Atmospheric component` and :ref:`files/model/oro_model:Mid-layer equations
    and the thermal wind relation` sections.

    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    delta_x: float, optional
        Spatial step in the zonal direction `x` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    delta_y: float, optional
        Spatial step in the meridional direction `y` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    dimensional: bool, optional
        Indicate if the output diagnostic must be dimensionalized or not.
        Default to `True`.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        self.type = "V"

        AtmosphericWindDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        self._plot_title = r'V wind in the middle of the atmosphere'

    def _get_diagnostic(self, dimensional):

        natm = self._model_params.nmod[0]
        psi = np.swapaxes(self._data[:natm, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)

        if dimensional:
            self._diagnostic_data = psi * self._model_params.streamfunction_scaling / self._model_params.scale_params.L
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = psi
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class MiddleAtmosphericUWindDiagnostic(AtmosphericWindDiagnostic):
    """Diagnostic giving the middle atmospheric U wind fields :math:`- \\partial_y \\psi_{\\rm a}` where
    :math:`\\psi_{\\rm a}` is the barotropic streamfunction.
    See also the :ref:`files/model/atmosphere:Atmospheric component` and :ref:`files/model/oro_model:Mid-layer equations
    and the thermal wind relation` sections.

    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    delta_x: float, optional
        Spatial step in the zonal direction `x` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    delta_y: float, optional
        Spatial step in the meridional direction `y` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    dimensional: bool, optional
        Indicate if the output diagnostic must be dimensionalized or not.
        Default to `True`.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        self.type = "U"

        AtmosphericWindDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        self._plot_title = r'U wind in the middle of the atmosphere'

    def _get_diagnostic(self, dimensional):

        natm = self._model_params.nmod[0]
        psi = np.swapaxes(self._data[:natm, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)

        if dimensional:
            self._diagnostic_data = - psi * self._model_params.streamfunction_scaling / self._model_params.scale_params.L
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = - psi
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class UpperLayerAtmosphericVWindDiagnostic(AtmosphericWindDiagnostic):
    """Diagnostic giving the upper layer atmospheric V wind fields :math:`\\partial_x \\psi^1_{\\rm a}`.
    Computed as :math:`\\partial_x \\psi^1_{\\rm a} = \\partial_x \\psi_{\\rm a} + \\partial_x \\theta_{\\rm a}` where :math:`\\psi_{\\rm a}` and :math:`\\theta_{\\rm a}` are respectively the barotropic and baroclinic streamfunctions.
    See also the :ref:`files/model/atmosphere:Atmospheric component` and :ref:`files/model/oro_model:Mid-layer equations and the thermal wind relation` sections.

    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    delta_x: float, optional
        Spatial step in the zonal direction `x` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    delta_y: float, optional
        Spatial step in the meridional direction `y` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    dimensional: bool, optional
        Indicate if the output diagnostic must be dimensionalized or not.
        Default to `True`.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        self.type = "V"

        AtmosphericWindDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        self._plot_title = r'Atmospheric V wind in the upper layer'

    def _get_diagnostic(self, dimensional):

        natm = self._model_params.nmod[0]
        psi = np.swapaxes(self._data[:natm, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)
        theta = np.swapaxes(self._data[natm:2*natm, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)

        psi1 = psi + theta

        if dimensional:
            self._diagnostic_data = psi1 * self._model_params.streamfunction_scaling / self._model_params.scale_params.L
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = psi1
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class UpperLayerAtmosphericUWindDiagnostic(AtmosphericWindDiagnostic):
    """Diagnostic giving the upper layer atmospheric U wind fields :math:`- \\partial_y \\psi^1_{\\rm a}`.
    Computed as :math:`- \\partial_y \\psi^1_{\\rm a} = - \\partial_y \\psi_{\\rm a} - \\partial_y \\theta_{\\rm a}` where :math:`\\psi_{\\rm a}` and :math:`\\theta_{\\rm a}` are respectively the barotropic and baroclinic streamfunctions.
    See also the :ref:`files/model/atmosphere:Atmospheric component` and :ref:`files/model/oro_model:Mid-layer equations and the thermal wind relation` sections.

    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    delta_x: float, optional
        Spatial step in the zonal direction `x` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    delta_y: float, optional
        Spatial step in the meridional direction `y` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    dimensional: bool, optional
        Indicate if the output diagnostic must be dimensionalized or not.
        Default to `True`.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        self.type = "U"

        AtmosphericWindDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        self._plot_title = r'Atmospheric U wind in the upper layer'

    def _get_diagnostic(self, dimensional):

        natm = self._model_params.nmod[0]
        psi = np.swapaxes(self._data[:natm, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)
        theta = np.swapaxes(self._data[natm:2*natm, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)

        psi1 = psi + theta

        if dimensional:
            self._diagnostic_data = - psi1 * self._model_params.streamfunction_scaling / self._model_params.scale_params.L
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = - psi1
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class LowerLayerAtmosphericWindIntensityDiagnostic(AtmosphericWindDiagnostic):
    """Diagnostic giving the lower layer atmospheric wind intensity fields.

    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    delta_x: float, optional
        Spatial step in the zonal direction `x` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    delta_y: float, optional
        Spatial step in the meridional direction `y` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    dimensional: bool, optional
        Indicate if the output diagnostic must be dimensionalized or not.
        Default to `True`.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        self.type = ""

        AtmosphericWindDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        self._plot_title = r'Atmospheric wind intensity in the lower layer'
        self._udiag = LowerLayerAtmosphericUWindDiagnostic(model_params, delta_x, delta_y, dimensional)
        self._vdiag = LowerLayerAtmosphericVWindDiagnostic(model_params, delta_x, delta_y, dimensional)

    def _get_diagnostic(self, dimensional):

        self._udiag.set_data(self._time, self._data)
        self._vdiag.set_data(self._time, self._data)

        U = self._udiag._get_diagnostic(dimensional)
        V = self._vdiag._get_diagnostic(dimensional)

        self._diagnostic_data = np.sqrt(U**2 + V**2)

        if dimensional:
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class MiddleAtmosphericWindIntensityDiagnostic(AtmosphericWindDiagnostic):
    """Diagnostic giving the middle atmospheric wind intensity fields.

    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    delta_x: float, optional
        Spatial step in the zonal direction `x` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    delta_y: float, optional
        Spatial step in the meridional direction `y` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    dimensional: bool, optional
        Indicate if the output diagnostic must be dimensionalized or not.
        Default to `True`.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        self.type = ""

        AtmosphericWindDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        self._plot_title = r'Wind intensity in the middle of the atmosphere'
        self._udiag = MiddleAtmosphericUWindDiagnostic(model_params, delta_x, delta_y, dimensional)
        self._vdiag = MiddleAtmosphericVWindDiagnostic(model_params, delta_x, delta_y, dimensional)

    def _get_diagnostic(self, dimensional):

        self._udiag.set_data(self._time, self._data)
        self._vdiag.set_data(self._time, self._data)

        U = self._udiag._get_diagnostic(dimensional)
        V = self._vdiag._get_diagnostic(dimensional)

        self._diagnostic_data = np.sqrt(U**2 + V**2)

        if dimensional:
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class UpperLayerAtmosphericWindIntensityDiagnostic(AtmosphericWindDiagnostic):
    """Diagnostic giving the lower layer atmospheric wind intensity fields.

    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    delta_x: float, optional
        Spatial step in the zonal direction `x` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    delta_y: float, optional
        Spatial step in the meridional direction `y` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    dimensional: bool, optional
        Indicate if the output diagnostic must be dimensionalized or not.
        Default to `True`.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        self.type = ""

        AtmosphericWindDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        self._plot_title = r'Atmospheric wind intensity in the upper layer'
        self._udiag = UpperLayerAtmosphericUWindDiagnostic(model_params, delta_x, delta_y, dimensional)
        self._vdiag = UpperLayerAtmosphericVWindDiagnostic(model_params, delta_x, delta_y, dimensional)

    def _get_diagnostic(self, dimensional):

        self._udiag.set_data(self._time, self._data)
        self._vdiag.set_data(self._time, self._data)

        U = self._udiag._get_diagnostic(dimensional)
        V = self._vdiag._get_diagnostic(dimensional)

        U = self._udiag._get_diagnostic(dimensional)
        V = self._vdiag._get_diagnostic(dimensional)

        self._diagnostic_data = np.sqrt(U**2 + V**2)

        if dimensional:
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


if __name__ == '__main__':
    from qgs.params.params import QgParams
    from qgs.integrators.integrator import RungeKuttaIntegrator
    from qgs.functions.tendencies import create_tendencies

    pars = QgParams()
    pars.set_atmospheric_channel_fourier_modes(2, 2)
    f, Df = create_tendencies(pars)
    integrator = RungeKuttaIntegrator()
    integrator.set_func(f)
    ic = np.random.rand(pars.ndim) * 0.1
    integrator.integrate(0., 200000., 0.1, ic=ic, write_steps=5)
    time, traj = integrator.get_trajectories()
    integrator.terminate()

    dx_psi3 = LowerLayerAtmosphericVWindDiagnostic(pars)
    dx_psi3(time, traj)

    dy_psi3 = LowerLayerAtmosphericUWindDiagnostic(pars)
    dy_psi3(time, traj)

    dx_psi = MiddleAtmosphericVWindDiagnostic(pars)
    dx_psi(time, traj)

    dy_psi = MiddleAtmosphericUWindDiagnostic(pars)
    dy_psi(time, traj)

    dx_psi1 = UpperLayerAtmosphericVWindDiagnostic(pars)
    dx_psi1(time, traj)

    dy_psi1 = UpperLayerAtmosphericUWindDiagnostic(pars)
    dy_psi1(time, traj)

    psi3_wind = LowerLayerAtmosphericWindIntensityDiagnostic(pars)
    psi3_wind(time, traj)

    psi_wind = MiddleAtmosphericWindIntensityDiagnostic(pars)
    psi_wind(time, traj)

    psi1_wind = UpperLayerAtmosphericWindIntensityDiagnostic(pars)
    psi1_wind(time, traj)
