# شرح وظيفة عمود الهدف (Target Column) في واجهة المستخدم

## نظرة عامة

عمود الهدف (Target Column) هو أحد المكونات الأساسية في نظام التحليل الطبي لدينا. يحدد هذا العمود المتغير الذي يحاول النموذج التنبؤ به أو تصنيفه. في سياق مجموعة البيانات الطبية لدينا، يكون عمود الهدف الافتراضي هو "خطر المرض" (disease_risk)، ولكن يمكن تغييره لدراسة العلاقات مع المؤشرات الحيوية الأخرى.

## لماذا يظهر "تشبع الأكسجين" (oxygen_saturation) كقيمة افتراضية؟

قد تلاحظ أن عمود "تشبع الأكسجين" (oxygen_saturation) يظهر أحيانًا كقيمة افتراضية في واجهة المستخدم. هذا يحدث للأسباب التالية:

1. **آلية الاكتشاف التلقائي**: عند تحميل ملف بيانات جديد، يحاول النظام تحديد عمود الهدف تلقائيًا. إذا لم يتمكن من العثور على عمود يحمل اسمًا واضحًا مثل "target" أو "label" أو "disease_risk"، فقد يختار آخر عمود في الملف.

2. **ترتيب الأعمدة**: في بعض ملفات البيانات، قد يكون "تشبع الأكسجين" هو آخر عمود قبل عمود "خطر المرض"، وفي حالة عدم وجود عمود "خطر المرض"، قد يتم اختياره افتراضيًا.

3. **الأهمية السريرية**: تشبع الأكسجين هو مؤشر حيوي مهم في تقييم المخاطر الصحية، خاصة للمشاكل القلبية والتنفسية، مما يجعله مرشحًا منطقيًا كمتغير هدف في بعض التحليلات.

## تأثير اختيار عمود الهدف على التحليل

يؤثر اختيار عمود الهدف بشكل كبير على كيفية عمل النموذج وتفسير النتائج:

### عند اختيار "خطر المرض" (disease_risk) كعمود هدف:

- **نوع المشكلة**: تصبح المشكلة تصنيفًا ثنائيًا (0 = خطر منخفض، 1 = خطر مرتفع).
- **هدف النموذج**: يتعلم النموذج العلاقات بين جميع المؤشرات الحيوية وخطر الإصابة بالمرض.
- **التفسير**: تمثل مخرجات النموذج احتمالية الإصابة بخطر المرض.
- **الاستخدام**: مناسب للفحص الطبي وتحديد المرضى المعرضين للخطر.

### عند اختيار مؤشر حيوي (مثل تشبع الأكسجين) كعمود هدف:

- **نوع المشكلة**: تصبح المشكلة تنبؤًا (انحدارًا) بقيمة رقمية.
- **هدف النموذج**: يتعلم النموذج كيفية التنبؤ بقيمة المؤشر الحيوي المحدد بناءً على المؤشرات الأخرى.
- **التفسير**: تمثل مخرجات النموذج القيمة المتوقعة للمؤشر الحيوي المحدد.
- **الاستخدام**: مفيد لدراسة العلاقات بين المؤشرات الحيوية أو تقدير قيم المؤشرات المفقودة.

## كيفية تغيير عمود الهدف

لتغيير عمود الهدف في واجهة المستخدم:

1. انتقل إلى صفحة "إدارة البيانات" (Dataset Management).
2. عند تحميل ملف بيانات جديد، ستجد حقل إدخال بعنوان "اسم عمود الهدف" (Target Column Name).
3. أدخل اسم العمود الذي ترغب في استخدامه كهدف (مثل "disease_risk" أو "glucose_level" أو أي عمود آخر موجود في البيانات).
4. اضغط على زر "تحميل ومعالجة" (Upload & Process).

## توصيات لاختيار عمود الهدف

- **للتنبؤ بخطر المرض**: استخدم "disease_risk" كعمود هدف.
- **لدراسة تأثير المؤشرات الحيوية على بعضها**: اختر أحد المؤشرات الحيوية كعمود هدف لفهم كيف تؤثر المؤشرات الأخرى عليه.
- **للتحليل الشامل**: قم بإجراء تجارب متعددة مع أعمدة هدف مختلفة لفهم العلاقات المتبادلة بين المؤشرات الحيوية.

## ملاحظة هامة

عند تغيير عمود الهدف، سيتم إعادة تدريب النموذج بالكامل وقد تختلف النتائج بشكل كبير. تأكد من اختيار العمود المناسب لأهداف تحليلك.
