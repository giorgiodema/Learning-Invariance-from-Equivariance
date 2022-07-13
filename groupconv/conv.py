import torch
from torch import nn
import math
from groupconv.groups import *

class LiftingKernelBase(torch.nn.Module):
    
    def __init__(self, group, kernel_size, in_channels, out_channels):
        """ 
        
        """
        super().__init__()
        self.group = group

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create spatial kernel grid. These are the coordinates on which our
        # kernel weights are defined.
        self.register_buffer("grid_R2", torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, self.kernel_size),
            torch.linspace(-1, 1, self.kernel_size),
        )).to(self.group.identity.device))

        # Transform the grid by the elements in this group.
        self.register_buffer("transformed_grid_R2", self.create_transformed_grid_R2())

    def create_transformed_grid_R2(self):
        """Transform the created grid by the group action of each group element.
        This yields a grid (over H) of spatial grids (over R2). In other words,
        a list of grids, each index of which is the original spatial grid transformed by
        a corresponding group element in H.
        
        """
        # Obtain all group elements.
        group_elements = self.group.elements()

        # Transform the grid defined over R2 with the sampled group elements.
        transformed_grid = self.group.left_action_on_R2(
            self.group.inverse(group_elements),
            self.grid_R2
        )
        return transformed_grid


    def sample(self, sampled_group_elements):
        """ Sample convolution kernels for a given number of group elements

        arguments should include:
        :param sampled_group_elements: the group elements over which to sample
            the convolution kernels

        should return:
        :return kernels: filter bank extending over all input channels, 
            containing kernels transformed for all output group elements.
        """
        raise NotImplementedError()


class InterpolativeLiftingKernel(LiftingKernelBase):

    def __init__(self, group, kernel_size, in_channels, out_channels):
        super().__init__(group, kernel_size, in_channels, out_channels)

        # create and initialise a set of weights, we will interpolate these
        # to create our transformed spatial kernels.
        self.weight = torch.nn.Parameter(torch.zeros((
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        ), device=self.group.identity.device))

        # Initialize weights using kaiming uniform intialisation
        torch.nn.init.kaiming_uniform_(self.weight.data, a=math.sqrt(5))
    
    def sample(self):
        """ Sample convolution kernels for a given number of group elements

        should return:
        :return kernels: filter bank extending over all input channels, 
            containing kernels transformed for all output group elements.
        """
        # We fold the output channel dim into the input channel dim; this allows
        # us to use the torch grid_sample function.
        weight = self.weight.view(
            1,
            self.out_channels * self.in_channels,
            self.kernel_size,
            self.kernel_size
        )

        # We want a transformed set of weights for each group element so 
        # we repeat the set of spatial weights along the output group axis
        weight = weight.repeat(self.group.elements().numel(), 1, 1, 1)

        # Sample the transformed kernels
        transformed_weight = torch.nn.functional.grid_sample(
            weight,
            self.transformed_grid_R2,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        # Separate input and output channels
        transformed_weight = transformed_weight.view(
            self.group.elements().numel(),
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        )

        # Put the output channel dimension before the output group dimension.
        transformed_weight = transformed_weight.transpose(0, 1)

        return transformed_weight


class LiftingConvolution(torch.nn.Module):

    def __init__(self, group, in_channels, out_channels, kernel_size,stride=1,padding=0,dilation=1):
        super().__init__()

        self.kernel = InterpolativeLiftingKernel(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels
        )
        self.stride = stride
        self.padding=padding
        self.dilation=dilation

    def forward(self, x):
        """ Perform lifting convolution

        @param x: Input sample [batch_dim, in_channels, spatial_dim_1, 
            spatial_dim_2]
        @return: Function on a homogeneous space of the group
            [batch_dim, out_channels, num_group_elements, spatial_dim_1, 
            spatial_dim_2]
        """

        # obtain convolution kernels transformed under the group
        conv_kernels = self.kernel.sample()

        # apply lifting convolution, note that the reshape folds the group 
        # dimension of the kernel into the output channel dimension.
        # Do you see why we (can) do this?
        x = torch.nn.functional.conv2d(
            input=x,
            weight=conv_kernels.reshape(
                self.kernel.out_channels * self.kernel.group.elements().numel(),
                self.kernel.in_channels,
                self.kernel.kernel_size,
                self.kernel.kernel_size
            ),
            stride = self.stride
        )

        # reshape [batch_dim, in_channels * num_group_elements, spatial_dim_1, 
        # spatial_dim_2] into [batch_dim, in_channels, num_group_elements, 
        # spatial_dim_1, spatial_dim_2], separating channel and group 
        # dimensions.
        x = x.view(
            -1,
            self.kernel.out_channels,
            self.kernel.group.elements().numel(),
            x.shape[-1],
            x.shape[-2]
        )

        return x


class GroupKernelBase(torch.nn.Module):

    def __init__(self, group, kernel_size, in_channels, out_channels):
        """ 
        
        """
        super().__init__()
        self.group = group

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create a spatial kernel grid
        self.register_buffer("grid_R2", torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, self.kernel_size),
            torch.linspace(-1, 1, self.kernel_size),
        )).to(self.group.identity.device))

        # The kernel grid now also extends over the group H, as our input 
        # feature maps contain an additional group dimension
        self.register_buffer("grid_H", self.group.elements())

        self.register_buffer("transformed_grid_R2xH", self.create_transformed_grid_R2xH())

    def create_transformed_grid_R2xH(self):
        """
        
        """
        # Sample the group
        group_elements = self.group.elements()

        # Transform the grid defined over R2 with the sampled group elements
        transformed_grid_R2 = self.group.left_action_on_R2(
            self.group.inverse(group_elements),
            self.grid_R2
        )

        # Transform the grid defined over H with the sampled group elements
        transformed_grid_H = self.group.left_action_on_H(
            self.group.inverse(group_elements), self.grid_H
        )

        # Rescale values to between -1 and 1, we do this to please the torch grid_sample
        # function.
        transformed_grid_H = self.group.normalize_group_elements(transformed_grid_H)

        # Create a combined grid as the product of the grids over R2 and H
        # repeat R2 along the group dimension, and repeat H along the spatial dimension
        # to create a [output_group_elem, num_group_elements, kernel_size, kernel_size, 3] grid
        transformed_grid = torch.cat(
            (
                transformed_grid_R2.view(
                    group_elements.numel(),
                    1,
                    self.kernel_size,
                    self.kernel_size,
                    2,
                ).repeat(1, group_elements.numel(), 1, 1, 1),
                transformed_grid_H.view(
                    group_elements.numel(),
                    group_elements.numel(),
                    1,
                    1,
                    1,
                ).repeat(1, 1, self.kernel_size, self.kernel_size, 1, )
            ),
            dim=-1
        )
        return transformed_grid


    def sample(self, sampled_group_elements):
        """ Sample convolution kernels for a given number of group elements

        arguments should include:
        :param sampled_group_elements: the group elements over which to sample
            the convolution kernels

        should return:
        :return kernels: filter bank extending over all input channels, 
            containing kernels transformed for all output group elements.
        """
        raise NotImplementedError()

class InterpolativeGroupKernel(GroupKernelBase):

    def __init__(self, group, kernel_size, in_channels, out_channels):
        super().__init__(group, kernel_size, in_channels, out_channels)

        # create and initialise a set of weights, we will interpolate these
        # to create our transformed spatial kernels. Note that our weight
        # now also extends over the group H
        self.weight = torch.nn.Parameter(torch.zeros((
            self.out_channels,
            self.in_channels,
            self.group.elements().numel(), # this is different from the lifting convolution
            self.kernel_size,
            self.kernel_size
        ), device=self.group.identity.device))

        # initialize weights using kaiming uniform intialisation
        torch.nn.init.kaiming_uniform_(self.weight.data, a=math.sqrt(5))
    
    def sample(self):
        """ Sample convolution kernels for a given number of group elements

        should return:
        :return kernels: filter bank extending over all input channels, 
            containing kernels transformed for all output group elements.
        """

        # fold the output channel dim into the input channel dim; this allows
        # us to use the torch grid_sample function
        weight = self.weight.view(
            1,
            self.out_channels * self.in_channels,
            self.group.elements().numel(),
            self.kernel_size,
            self.kernel_size
        )

        # we want a transformed set of weights for each group element so 
        weight = weight.repeat(self.group.elements().numel(), 1, 1, 1, 1)

        # sample the transformed kernels, 
        transformed_weight = torch.nn.functional.grid_sample(
            weight,
            self.transformed_grid_R2xH,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        # Separate input and output channels. Note we now have a notion of
        # input and output group dimensions in our weight matrix!
        transformed_weight = transformed_weight.view(
            self.group.elements().numel(), # Output group elements (like in the lifting convolutoin)
            self.out_channels,
            self.in_channels,
            self.group.elements().numel(), # Input group elements (due to the additional dimension of our feature map)
            self.kernel_size,
            self.kernel_size
        )

        # Put the output channel dimension before the output group dimension.
        transformed_weight = transformed_weight.transpose(0, 1)

        return transformed_weight

    
class GroupConvolution(torch.nn.Module):

    def __init__(self, group, in_channels, out_channels, kernel_size,stride=1,padding=0,dilation=1):
        super().__init__()

        self.kernel = InterpolativeGroupKernel(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels
        )
        self.stride=stride
        self.padding=padding
        self.dilation=dilation

    def forward(self, x):
        """ Perform lifting convolution

        @param x: Input sample [batch_dim, in_channels, group_dim, spatial_dim_1, 
            spatial_dim_2]
        @return: Function on a homogeneous space of the group
            [batch_dim, out_channels, num_group_elements, spatial_dim_1, 
            spatial_dim_2]
        """

        # We now fold the group dimensions of our input into the input channel
        # dimension
        x = x.reshape(
            -1,
            x.shape[1] * x.shape[2],
            x.shape[3],
            x.shape[4]
        )

        # We obtain convolution kernels transformed under the group
        conv_kernels = self.kernel.sample()

        # Apply group convolution, note that the reshape folds the group 
        # dimension of the kernel into the output channel dimension.
        # Do you see why we (can) do this?
        x = torch.nn.functional.conv2d(
            input=x,
            weight=conv_kernels.reshape(
                self.kernel.out_channels * self.kernel.group.elements().numel(),
                self.kernel.in_channels * self.kernel.group.elements().numel(),
                self.kernel.kernel_size,
                self.kernel.kernel_size
            ),
            stride=self.stride,
            padding=self.padding,
            dilation = self.dilation
        )

        # Reshape [batch_dim, in_channels * num_group_elements, spatial_dim_1, 
        # spatial_dim_2] into [batch_dim, in_channels, num_group_elements, 
        # spatial_dim_1, spatial_dim_2], separating channel and group 
        # dimensions.
        x = x.view(
            -1,
            self.kernel.out_channels,
            self.kernel.group.elements().numel(),
            x.shape[-1],
            x.shape[-2],
        )

        return x


class GroupAvgPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        # average the group dimension
        o = torch.mean(input,dim=2)
        return o

class GroupMaxPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        # average the group dimension
        o = torch.max(input,dim=2).values
        return o

class SpatialGlobalAvgPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        # average the spatial dimension
        o = torch.mean(input,dim=[-1,-2])
        return o

class GroupBatchNorm2d(nn.BatchNorm2d):
    def __init__(self,group_actions, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__(num_features * group_actions, eps, momentum, affine, track_running_stats, device, dtype)

    def forward(self, input):
        group_el = input.shape[1]
        nfilters = input.shape[2]
        input = torch.reshape(input,(input.shape[0],group_el*nfilters,input.shape[-2],input.shape[-1]))
        o =  super().forward(input)
        o = torch.reshape(o,(o.shape[0],group_el,nfilters,o.shape[-2],o.shape[-1]))
        return o

class SpatialMaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride = None, padding = 0, dilation = 1, return_indices: bool = False, ceil_mode: bool = False) -> None:
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    def forward(self, input):
        group_el = input.shape[1]
        nfilters = input.shape[2]
        input = torch.reshape(input,(input.shape[0],group_el*nfilters,input.shape[-2],input.shape[-1]))
        o =  super().forward(input)
        o = torch.reshape(o,(o.shape[0],group_el,nfilters,o.shape[-2],o.shape[-1]))
        return o

class SpatialAvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size, stride = None, padding = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override = None) -> None:
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    def forward(self, input):
        group_el = input.shape[1]
        nfilters = input.shape[2]
        input = torch.reshape(input,(input.shape[0],group_el*nfilters,input.shape[-2],input.shape[-1]))
        o =  super().forward(input)
        o = torch.reshape(o,(o.shape[0],group_el,nfilters,o.shape[-2],o.shape[-1]))
        return o

class GlobalMaxPooling(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x):
        o = torch.max(x,dim=-1).values
        o = torch.max(o,dim=-1).values
        return o