from typing import TypedDict

from .loss import (calculate_loss_metrics,
                   frequency_loss,
                   anharmonicity_loss,
                   anharmonicity_loss_constantnorm,
                   flux_sensitivity_loss,
                   flux_sensitivity_loss_constantnorm,
                   charge_sensitivity_loss,
                   T1_loss)


class LossFunctions(TypedDict):
    omega: str
    aharm: str
    T1: str
    flux: str
    charge: str
loss1: TypedDict = {
    'omega':frequency_loss,
    'aharm': anharmonicity_loss,
    'T1': T1_loss,
    'flux': flux_sensitivity_loss,
    'charge': charge_sensitivity_loss
}
loss2: TypedDict = {
    'omega':frequency_loss,
    'aharm': anharmonicity_loss_constantnorm,
    'T1': T1_loss,
    'flux': flux_sensitivity_loss_constantnorm,
    'charge': charge_sensitivity_loss
}

loss_functions = {
    'default': lambda cr, **kwargs: calculate_loss_metrics(cr, function_dict=loss1, **kwargs),
    'constant_norm': lambda cr, **kwargs: calculate_loss_metrics(cr, function_dict=loss2, **kwargs)
}
