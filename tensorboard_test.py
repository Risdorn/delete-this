from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs')
writer.add_text('example_text', 'This is an example text for TensorBoard.')
print("TensorBoard test completed. Check the 'runs' directory for logs.")