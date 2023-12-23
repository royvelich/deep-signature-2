import torch


def calculate_pearson_loss_vectorized(matrix, device):
    # Convert the matrix to a PyTorch tensor
    # matrix = torch.tensor(matrix)

    # Get the number of columns in the matrix
    num_columns = matrix.size(1)

    # Calculate the Pearson correlation coefficient for each pair of columns
    # correlation_matrix = torch.nn.functional.cosine_similarity(matrix.T.unsqueeze(0), matrix.T.unsqueeze(1), dim=2)
    correlation_matrix = torch.corrcoef(matrix.T)

    nan_mask = torch.isnan(correlation_matrix)
    # Replace NaN elements with 0
    correlation_matrix[nan_mask] = 0

    # Create an identity matrix
    identity = torch.eye(num_columns)
    identity[0,1] = 0.5
    identity[1,0] = 0.5
    # Define the non-zero entries positions in the identity matrix
    non_zero_entries = [(0, 6), (1, 7)]

    # Create a binary mask for the non-zero entries
    mask = torch.ones_like(identity)
    for i, j in non_zero_entries:
        mask[i, j] = 0
        mask[j, i] = 0

    # Apply the mask to the correlation matrix

    mask= mask.to(device)
    correlation_matrix=correlation_matrix.to(device)
    identity=identity.to(device)

    correlation_matrix *= mask

    # Calculate the loss
    loss = torch.linalg.matrix_norm(identity - correlation_matrix) ** 2
    # print("pearson loss:", loss)

    return loss

def calculate_pearson_k1_k2_loss_vectorized(matrix, device):


    # Get the number of columns in the matrix
    num_columns = matrix.size(1)

    # correlation_matrix = torch.nn.functional.cosine_similarity(matrix.T.unsqueeze(0), matrix.T.unsqueeze(1), dim=2)
    correlation_matrix = torch.corrcoef(matrix.T)

    # nan_mask = torch.isnan(correlation_matrix)
    # # Replace NaN elements with 0
    # correlation_matrix[nan_mask] = 0

    # Create an identity matrix
    identity = torch.eye(num_columns)
    identity[0,1] = 0.5
    identity[1,0] = 0.5


    # Apply the mask to the correlation matrix

    identity= identity.to(device)
    correlation_matrix=correlation_matrix.to(device)


    # Calculate the loss
    loss = torch.linalg.matrix_norm(identity - correlation_matrix) ** 2
    # print("pearson loss:", loss)

    return loss

def codazzi_loss(output):
    """

    :param output: matrix of size (n, 8) where n is the number of vertices in a batch
    :return:
    """
#     k^1_22-k^2_11 + (k^1_1k^2_1+k^1_2k^2_2-2(k^2_1)^2-2(k^1_2)^2)/(k^1-k^2)- k^1k^2(k^1-k^2) = 0

    k1 = output[:,0]
    k2 = output[:,1]
    k1_der1 = output[:,2]
    k2_der1 = output[:,3]
    k1_der2 = output[:,4]
    k2_der2 = output[:,5]
    k1_der22 = output[:,6]
    k2_der11 = output[:,7]
    loss = (torch.norm((k1_der22 - k2_der11) + ((k1_der1*k2_der1 + k1_der2*k2_der2 - 2*k2_der1**2 - 2*k1_der2**2)/(abs(k1-k2)+1e-4))*torch.sign(k1-k2) - k1*k2*(k1-k2)))/output.size(0)
    # print("codazzi loss:", loss)

    # return (torch.norm((k1_der22 - k2_der11)*(k1-k2) + (k1_der1*k2_der1 + k1_der2*k2_der2 - 2*k2_der1**2 - 2*k1_der2**2) - k1*k2*((k1-k2)**2))**2)
    return loss
    # return (torch.norm((k1_der22 - k2_der11)*(k1-k2) + (k1_der1*k2_der1 + k1_der2*k2_der2 - 2*k2_der1**2 - 2*k1_der2**2) - k1*k2*((k1-k2)**2))**2)/output.size(0)


def sanity_check_pc():
    # input_matrix = torch.tensor([
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9],
    #     [10, 11, 12]
    # ],dtype=torch.float)

    # input_matrix = torch.randn(4, 3)
    # input_matrix = torch.ones(4, 3)
    input_matrix = torch.eye(4)
    print(input_matrix)

    # Calculate the correlation matrix
    correlation_matrix = calculate_pearson_loss_vectorized(input_matrix)

    # Print the correlation matrix
    print(correlation_matrix)

# sanity_check_pc()

def contrastive_tuplet_loss(a,p,n):
    loss = torch.log(1 + torch.exp(torch.linalg.matrix_norm(a-p)**2 - torch.linalg.matrix_norm(a-n)**2))/a.size(0)
    # print("contrastive loss:", loss)
    if torch.isnan(loss):
        print("loss is NaN")
    return loss
    # return torch.log(1 + torch.exp(torch.linalg.matrix_norm(a-p)**2 - torch.linalg.matrix_norm(a-n)**2))/a.size(0)

def loss_contrastive_plus_pc(a,p,n):
    return (1/a.size(0))*(contrastive_tuplet_loss(a,p,n) + calculate_pearson_loss_vectorized(torch.stack([a,p,n], dim=0)))

def loss_contrastive_plus_codazzi(a,p,n):
    return (contrastive_tuplet_loss(a,p,n) + 0.01*codazzi_loss(torch.cat([a,p,n], dim=0)))

def loss_contrastive_plus_codazzi_and_pearson_correlation(a,p,n, device='cpu'):
    return (contrastive_tuplet_loss(a,p,n) + codazzi_loss(torch.cat([a,p,n], dim=1).T)+0.01*calculate_pearson_loss_vectorized(torch.cat([a,p,n], dim=1).T, device))

def loss_contrastive_plus_codazzi_and_pearson_correlation_k1_k2(a,p,n, device='cpu'):
    return (contrastive_tuplet_loss(a,p,n) + 0.1*calculate_pearson_k1_k2_loss_vectorized(torch.cat([a,p,n], dim=1).T, device))


def loss_codazzi_and_pearson_correlation(output, device='cpu'):
    return (codazzi_loss(output)+0.01*calculate_pearson_loss_vectorized(output, device))

def loss__pearson_correlation_k1_k2(a,p,n, device='cpu'):
    return (calculate_pearson_k1_k2_loss_vectorized(torch.cat([a,p,n], dim=1).T, device))

def loss_contrastive_plus_pearson_correlation_k1_k2(a,p,n, device='cpu'):
    return (contrastive_tuplet_loss(a,p,n) +0.2*calculate_pearson_k1_k2_loss_vectorized(torch.cat([a,p,n], dim=1).T, device))

def loss_gaussian_curvature_supervised(output, principal_curvatures):
    return (1/output.size(0))*torch.norm(output[:,0]*output[:,1] - principal_curvatures[0]*principal_curvatures[1])**2