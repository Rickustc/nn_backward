class Tensor:
    def __init__(self, value, requires_grad=False, name=None):
        self.value = float(value)
        self.requires_grad = requires_grad
        self.grad = 0.0
        self.grad_fn = None
        self.is_leaf = True
        self.name = name if name is not None else "tensor"

    def backward(self, grad_output=1.0):
        """
        1. 如果当前 tensor 不需要梯度，直接返回
        2. 如果是 leaf / 没有 grad_fn，则把梯度累加到自己
        3. 否则调用 grad_fn.backward(...)，再把返回的梯度递归传给输入
        """
        if not self.requires_grad:
            print(f"[Skip backward] {self.name} does not require grad")
            return

        if self.grad_fn is None:
            self.grad += grad_output
            print(
                f"[Leaf backward] {self.name}.grad += {grad_output:.4f} "
                f"-> {self.grad:.4f}"
            )
            return

        print(f"[Backward enter] tensor={self.name}, grad_output={grad_output:.4f}")
        grads = self.grad_fn.backward(grad_output)

        for inp, g in zip(self.grad_fn.inputs, grads):
            if isinstance(inp, Tensor) and inp.requires_grad:
                inp.backward(g)

    def zero_grad(self):
        self.grad = 0.0

    def grad_fn_name(self):
        if self.grad_fn is None:
            return None
        return self.grad_fn.cls.__name__

    def summary(self):
        return (
            f"Tensor(name={self.name}, value={self.value:.4f}, "
            f"requires_grad={self.requires_grad}, grad={self.grad:.4f}, "
            f"grad_fn={self.grad_fn_name()})"
        )

    def __repr__(self):
        return self.summary()


class Context:
    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class BackwardNode:
    """
    表示“一次具体的 Function 调用”对应的 backward 节点
    """
    def __init__(self, cls, ctx, inputs):
        self.cls = cls
        self.ctx = ctx
        self.inputs = inputs

    def backward(self, grad_output):
        print(f"[Node backward] call {self.cls.__name__}.backward(grad_output={grad_output:.4f})")
        return self.cls.backward(self.ctx, grad_output)


class Function:
    @classmethod
    def apply(cls, *inputs):
        """
        1. 创建这次 forward 专属的 ctx
        2. 调用户写的 forward
        3. 若输入中有 tensor 需要梯度，则创建 backward node
        4. 把 backward node 挂到输出 tensor 的 grad_fn 上
        """
        ctx = Context()

        input_desc = ", ".join(
            f"{x.name}={x.value:.4f}" if isinstance(x, Tensor) else str(x)
            for x in inputs
        )
        print(f"\n[Apply] {cls.__name__}.apply({input_desc})")

        output = cls.forward(ctx, *inputs)

        if not isinstance(output, Tensor):
            raise TypeError("In this teaching demo, forward must return a Tensor.")

        requires_grad = any(
            isinstance(x, Tensor) and x.requires_grad for x in inputs
        )

        if requires_grad:
            node = BackwardNode(cls=cls, ctx=ctx, inputs=inputs)
            output.requires_grad = True
            output.is_leaf = False
            output.grad_fn = node
            print(
                f"[Apply] attach grad_fn={cls.__name__} to output {output.name}"
            )
        else:
            print(f"[Apply] output {output.name} has no grad_fn")

        print(f"[Apply] output summary: {output.summary()}")
        return output


class MyMul(Function):
    @staticmethod
    def forward(ctx, left, right):
        print(f"[Forward] MyMul: {left.name} * {right.name}")
        ctx.save_for_backward(left, right)
        out = Tensor(
            left.value * right.value,
            requires_grad=False,
            name=f"({left.name}*{right.name})"
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        left, right = ctx.saved_tensors
        grad_left = grad_output * right.value
        grad_right = grad_output * left.value

        print(
            f"[Backward] MyMul for z={left.name}*{right.name}: "
            f"grad_left = {grad_output:.4f} * {right.value:.4f} = {grad_left:.4f}, "
            f"grad_right = {grad_output:.4f} * {left.value:.4f} = {grad_right:.4f}"
        )
        return grad_left, grad_right


class MyAdd(Function):
    @staticmethod
    def forward(ctx, left, right):
        print(f"[Forward] MyAdd: {left.name} + {right.name}")
        ctx.save_for_backward(left, right)
        out = Tensor(
            left.value + right.value,
            requires_grad=False,
            name=f"({left.name}+{right.name})"
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        left, right = ctx.saved_tensors

        grad_left = grad_output
        grad_right = grad_output

        print(
            f"[Backward] MyAdd for z={left.name}+{right.name}: "
            f"grad_left = {grad_left:.4f}, grad_right = {grad_right:.4f}"
        )
        return grad_left, grad_right


def mul(left, right):
    return MyMul.apply(left, right)


def add(left, right):
    return MyAdd.apply(left, right)



if __name__ == "__main__":
    a = Tensor(2.0, requires_grad=True, name="a")
    x = Tensor(3.0, requires_grad=True, name="x")
    b = Tensor(4.0, requires_grad=True, name="b")

    print("===== Initial Tensors =====")
    print(a.summary())
    print(x.summary())
    print(b.summary())

    print("\n===== Forward: y = a*x + b =====")
    ax = mul(a, x)
    y = add(ax, b)

    print("\n===== Tensor States After Forward =====")
    print(a.summary())
    print(x.summary())
    print(b.summary())
    print(ax.summary())
    print(y.summary())

    print("\n===== Backward: compute dy/d(...) =====")
    y.backward(1.0)

    print("\n===== Final Gradients =====")
    print(f"dy/da = {a.grad:.4f}")
    print(f"dy/dx = {x.grad:.4f}")
    print(f"dy/db = {b.grad:.4f}")

    print("\n===== Analytic Check =====")
    print("For y = a*x + b:")
    print("dy/da = x")
    print("dy/dx = a")
    print("dy/db = 1")
    print(f"Current values: a={a.value:.4f}, x={x.value:.4f}, b={b.value:.4f}")
    print(f"Expected: dy/da={x.value:.4f}, dy/dx={a.value:.4f}, dy/db=1.0000")