import torch

def impute_missing_data(data: torch.Tensor, strategy: str = 'mean') -> torch.Tensor:
    """
    Impute missing values in a 2D tensor using the specified strategy.

    Args:
        data: 2D torch.Tensor with missing values represented as float('nan')
        strategy: Imputation strategy - 'mean', 'median', or 'mode'

    Returns:
        2D torch.Tensor with missing values imputed
    """
    """
        first time i thought about nested loop

        row,col=data.shape
        for i in range(row):
            for j in range(col):
                if torch.isnan(data[i,j]):
                    col=data[:,j]
                    col_mean=col[~torch.isnan(col)].mean()
                    data[i,j]=col_mean
        
        but is already filter function
    """
    row,col=data.shape
    is_nan=torch.isnan(data)
    mean_col=torch.nanmean(data,dim=0,keepdim=True)
    mediean_col=torch.nanmedian(data,dim=0,keepdim=True).values
    fills=[]
    if strategy=='mean':
        return torch.where(is_nan,mean_col,data)
    elif strategy=='median':
        return torch.where(is_nan,mediean_col,data)
    else:
        for j in range(col):
            col=data[:,j]
            valid=col[~torch.isnan(col)]
            u,c=torch.unique(valid,return_counts=True)
            fills.append(u[torch.argmax(c)])
        mode_col=torch.stack(fills).unsqueeze(0)
        return torch.where(is_nan,mode_col,data)

