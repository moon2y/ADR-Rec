import torch
import torch.nn as nn
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class DomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 10)
        self.fc2 = nn.Linear(10, 3)  # domain classes

    def forward(self, x, reverse=False):
        if reverse:
            x = GradReverse.apply(x)
        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x)

class DANN_BERT4Rec(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.specific = base_model
        self.common = base_model
        self.domain_classifier = DomainClassifier()

    def forward(self, log_seqs):
        logits_s, out_s = self.specific(log_seqs)
        logits_c, out_c = self.common(log_seqs)

        domain_logits_s = self.domain_classifier(out_s, reverse=False)
        domain_logits_c = self.domain_classifier(out_c, reverse=True)

        return logits_s, logits_c, domain_logits_s, domain_logits_c

    def extract_embeddings(self, log_seqs):
        _, out_s = self.specific(log_seqs)
        _, out_c = self.common(log_seqs)
        return out_s, out_c

class DANN_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MCE = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, outputs, labels, domain_id, alpha):
        logits_s, logits_c, domain_logits_s, domain_logits_c = outputs
        logits_s = logits_s.view(-1, logits_s.size(-1))
        logits_c = logits_c.view(-1, logits_c.size(-1))
        labels = labels.view(-1)

        batch_size = domain_logits_s.size(0)
        domain_targets = torch.LongTensor([domain_id] * batch_size).to(labels.device)

        domain_loss_s = self.CE(domain_logits_s, domain_targets)
        domain_loss_c = self.CE(domain_logits_c, domain_targets)

        masked_loss_s = self.MCE(logits_s, labels)
        masked_loss_c = self.MCE(logits_c, labels)

        loss_s = masked_loss_s + alpha * domain_loss_s
        loss_c = masked_loss_c + alpha * domain_loss_c

        return loss_s, loss_c
